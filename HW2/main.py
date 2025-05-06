import torch
from tqdm import tqdm
import cv2
import heapq
import argparse
def make_gaussian_kernel(size, sigma, device):
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def extract_patches_torch(sample_tensor, window_size):
    """Extract RGB patches as batched tensor from [3, H, W]."""
    C, H, W = sample_tensor.shape
    patches = sample_tensor.unfold(1, window_size, 1).unfold(2, window_size, 1)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, window_size, window_size)

    half_w = window_size // 2
    centers = sample_tensor[:, half_w:H-half_w, half_w:W-half_w].permute(1, 2, 0).reshape(-1, C)
    return patches, centers

def count_known_neighbors(mask, y, x, window_size):
    y_start = max(0, y - window_size//2)
    y_end = min(mask.shape[0], y + window_size//2 + 1)
    x_start = max(0, x - window_size//2)
    x_end = min(mask.shape[1], x + window_size//2 + 1)
    return mask[y_start:y_end, x_start:x_end].sum().item()

def efros_leung_synthesis_cuda(sample_patches, sample_centers, out_size=128,
                              window_size=11, seed_size=3, device='cuda'):
    """Efros-Leung synthesis with RGB image and CUDA acceleration."""
    C = 3
    out_img = torch.zeros((C, out_size, out_size), dtype=torch.float32, device=device)
    synthesized = torch.zeros((out_size, out_size), dtype=torch.bool, device=device)

    # Seed
    seed_y = torch.randint(0, out_size - seed_size, (1,)).item()
    seed_x = torch.randint(0, out_size - seed_size, (1,)).item()
    out_img[:, seed_y:seed_y+seed_size, seed_x:seed_x+seed_size] = 0.5
    synthesized[seed_y:seed_y+seed_size, seed_x:seed_x+seed_size] = True

    # Gaussian kernel
    gaussian = make_gaussian_kernel(window_size, sigma=window_size//6, device=device)
    gaussian = gaussian.unsqueeze(0).expand(C, -1, -1)  # [3, w, w]

    # Prepare patch database
    sample_patches = sample_patches.to(device)
    sample_centers = sample_centers.to(device)

    half_w = window_size // 2
    q = []
    for y in range(seed_y, seed_y + seed_size):
        for x in range(seed_x, seed_x + seed_size):
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < out_size and 0 <= nx < out_size and not synthesized[ny, nx]:
                    heapq.heappush(q, (-count_known_neighbors(synthesized, ny, nx, window_size), (ny, nx)))

    progress = tqdm(total=out_size*out_size - seed_size**2)
    while q:
        _, (y, x) = heapq.heappop(q)
        if synthesized[y, x]:
            continue

        y_start = max(0, y - half_w)
        y_end = min(out_size, y + half_w + 1)
        x_start = max(0, x - half_w)
        x_end = min(out_size, x + half_w + 1)

        window = out_img[:, y_start:y_end, x_start:x_end]
        mask = synthesized[y_start:y_end, x_start:x_end].float()

        pad_top = half_w - (y - y_start)
        pad_bottom = half_w - (y_end - y - 1)
        pad_left = half_w - (x - x_start)
        pad_right = half_w - (x_end - x - 1)

        window = torch.nn.functional.pad(window, (pad_left, pad_right, pad_top, pad_bottom))
        mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom))
        mask = mask.unsqueeze(0).expand_as(window)

        diff = (sample_patches - window.unsqueeze(0)) * mask.unsqueeze(0)
        errors = ((diff ** 2) * gaussian).sum(dim=(1, 2, 3)) / mask.sum()

        min_error = errors.min()
        valid = errors <= min_error * 1.2
        candidates = sample_centers[valid]

        if len(candidates) > 0:
            out_img[:, y, x] = candidates[torch.randint(len(candidates), (1,))]
            synthesized[y, x] = True
            progress.update(1)

            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < out_size and 0 <= nx < out_size and not synthesized[ny, nx]:
                    heapq.heappush(q, (-count_known_neighbors(synthesized, ny, nx, window_size), (ny, nx)))

    progress.close()
    return (out_img.clamp(0,1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

def efros_leung_synthesis_with_one_image(image_path, out_size, window_size):
    # Load RGB sample image
    sample = cv2.imread(image_path)  # BGR to RGB
    sample = torch.tensor(sample, dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255.0  # [3, H, W]

    # Extract patches
    window_size = 11
    patches, centers = extract_patches_torch(sample, window_size)

    # Synthesize
    synthesized = efros_leung_synthesis_cuda(
        patches, centers,
        out_size=out_size,
        window_size=window_size
    )

    return synthesized

def extract_patches_from_mnist(dataset, window_size=7, device='cuda', max_images=20000):
    all_patches = []
    all_centers = []
    for idx in range(min(max_images, len(dataset))):
        img, _ = dataset[idx]  # [1, 28, 28]
        img = img.to(device)  # [28, 28]
        patches, centers = extract_patches_torch(img, window_size)
        all_patches.append(patches)
        all_centers.append(centers)
    
    all_patches = torch.cat(all_patches, dim=0)
    all_centers = torch.cat(all_centers, dim=0)
    return all_patches, all_centers

def run_tenuer_experiment():
    # First Experiment
    for image_id in range(1, 4):
        image_path = f'./images/{image_id}.png'
        synthesized_path = f'./output/regular_textures/synthesized_{image_id}.png'
        synthesized = efros_leung_synthesis_with_one_image(image_path=image_path, out_size=1024, window_size=7)
        cv2.imwrite(synthesized_path, synthesized)

def run_MNIST_experiment():
    # Second Experiment
    from torchvision.datasets import MNIST
    from torchvision import transforms

    transform = transforms.ToTensor()
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)

    window_sizes = [5, 7, 11]

    for window_size in window_sizes:
        patches, centers = extract_patches_from_mnist(mnist_train, window_size=window_size, device='cuda')
        synthesized = efros_leung_synthesis_cuda(patches, centers, out_size=128, window_size=window_size, device='cuda')
        cv2.imwrite(f'./output/mnist/synthesized_{window_size}.png', synthesized)

def main(args):
    if args.run_tenuer_experiment:
        run_tenuer_experiment()

    if args.run_MNIST_experiment:
        run_MNIST_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tenuer_experiment", action="store_true")
    parser.add_argument("--run_MNIST_experiment", action="store_true")
    args = parser.parse_args()

    main(args)