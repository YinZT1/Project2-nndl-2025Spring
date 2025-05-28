# task1_train_vgg16.py (修改版)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os

from models.vgg16 import VGG 

# --- Helper function to save checkpoint ---
def save_checkpoint(state, is_best, filename_prefix='checkpoint', checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, f'{filename_prefix}.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f'{filename_prefix}_best.pth.tar')
        torch.save(state, best_filename) # 直接保存 best 状态，而非复制
        print(f"Epoch {state['epoch']}: Best model saved to {best_filename} with accuracy {state['best_acc']:.4f}")

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VGG Training with Configurations')
    parser.add_argument('--vgg_config', default='VGG16_standard', type=str, help='VGG configuration name (e.g., VGG16_standard, VGG16_light)')
    parser.add_argument('--activation', default='relu', type=str, help='Activation function (relu, leaky_relu, tanh)')
    parser.add_argument('--loss_fn', default='cross_entropy', type=str, help='Loss function (cross_entropy, nll_loss)')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer (adam, sgd, adamw)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for AdamW or SGD')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs') # 增加默认epoch
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loader')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_cifar10', type=str, help='Directory to save checkpoints')
    parser.add_argument('--run_name', default='vgg_run', type=str, help='A name for this specific run/experiment')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Reproducibility and Device ---
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True # Might impact performance slightly
        torch.backends.cudnn.benchmark = False

    print(f"Using device: {device}")
    print("Current Hyperparameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 30)


    # --- Data Loading and Preprocessing ---
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # Not used directly here

    # --- Model ---
    print('==> Building model..')
    use_log_softmax_for_model = (args.loss_fn == 'nll_loss')
    model = VGG(vgg_name=args.vgg_config, activation_name=args.activation, num_classes=10, use_log_softmax=use_log_softmax_for_model)
    model = model.to(device)

    # --- Loss Function ---
    if args.loss_fn == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == 'nll_loss':
        criterion = nn.NLLLoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")

    # --- Optimizer ---
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay if args.weight_decay > 0 else 0)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay if args.weight_decay > 0 else 1e-2) # AdamW usually needs weight_decay
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Learning rate scheduler (optional, but good for SGD)
    scheduler = None
    if args.optimizer == 'sgd':
         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)


    # --- Training ---
    start_epoch = 0
    best_acc = 0.0

    # You might want to add resume from checkpoint logic here if needed

    print(f"==> Starting Training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0 or batch_idx == len(trainloader) -1 :
                print(f'Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx+1}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} training completed in {epoch_time:.2f}s")

        # --- Validation ---
        model.eval()
        test_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_test_loss = test_loss / len(testloader)
        print(f'Epoch {epoch+1} Validation: Loss: {avg_test_loss:.3f} | Acc: {val_acc:.3f}% ({val_correct}/{val_total})')

        # --- Save checkpoint ---
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint_filename_prefix = f"{args.run_name}_vgg_{args.vgg_config}_act_{args.activation}_loss_{args.loss_fn}_opt_{args.optimizer}_lr_{args.lr}"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'args': args # Save args for reference
        }, is_best, filename_prefix=checkpoint_filename_prefix, checkpoint_dir=args.checkpoint_dir)

        if scheduler:
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")


    print(f"Finished Training. Best Validation Accuracy: {best_acc:.3f}%")
    print(f"Find checkpoints in: {args.checkpoint_dir}")

if __name__ == '__main__':
    main()