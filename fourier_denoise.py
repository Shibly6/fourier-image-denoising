import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fourier_denoise(image_path, output_path, threshold_percentage):

    # Load the noisy image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)

    # Calculate magnitude spectrum (for visualization)
    magnitude_spectrum = np.abs(f_shift)

    # Calculate threshold based on magnitude
    sorted_magnitudes = np.sort(magnitude_spectrum.flatten())
    threshold_idx = int((1 - threshold_percentage) * len(sorted_magnitudes))
    threshold = sorted_magnitudes[threshold_idx]

    # Create mask - keep only the significant frequencies
    mask = magnitude_spectrum > threshold

    # Apply mask to the Fourier transform
    f_shift_filtered = f_shift * mask

    # Inverse Fourier Transform to get back to spatial domain
    f_inverse_shift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_inverse_shift)
    img_back = np.abs(img_back)

    # Normalize to 0-255 and convert to uint8
    img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255).astype(np.uint8)

    # Create a PIL image from the denoised array
    denoised_img = Image.fromarray(img_back)

    # Save the denoised image
    denoised_img.save(output_path)

    # Create 3D representation of the magnitude spectrum
    X = np.linspace(-img_array.shape[1] // 2, img_array.shape[1] // 2, img_array.shape[1])
    Y = np.linspace(-img_array.shape[0] // 2, img_array.shape[0] // 2, img_array.shape[0])
    X, Y = np.meshgrid(X, Y)

    # Enhanced visualization with detailed subplots
    plt.figure(figsize=(18, 12))

    # Create a complex grid layout
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Original noisy image
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original Noisy Image', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    # Magnitude spectrum (log scale)
    ax2 = plt.subplot(gs[0, 1])
    # Use log scale for better visualization of the spectrum
    spectrum_log = np.log1p(magnitude_spectrum)
    im2 = ax2.imshow(spectrum_log, cmap='viridis', norm=LogNorm(vmin=0.01, vmax=spectrum_log.max()))
    ax2.set_title('Fourier Transform Magnitude Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    # Filtered magnitude spectrum
    ax3 = plt.subplot(gs[0, 2])
    filtered_spectrum_log = np.log1p(magnitude_spectrum * mask)
    im3 = ax3.imshow(filtered_spectrum_log, cmap='viridis', norm=LogNorm(vmin=0.01, vmax=spectrum_log.max()))
    ax3.set_title(f'Filtered Spectrum (Top {threshold_percentage * 100:.1f}% Frequencies)', fontsize=12,
                  fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)

    # 3D visualization of the spectrum
    ax4 = plt.subplot(gs[1, 0], projection='3d')
    # Downsample for performance if needed
    downsample = 4
    X_down = X[::downsample, ::downsample]
    Y_down = Y[::downsample, ::downsample]
    Z_down = np.log1p(magnitude_spectrum)[::downsample, ::downsample]

    surf = ax4.plot_surface(X_down, Y_down, Z_down, cmap='viridis', linewidth=0,
                            antialiased=True, alpha=0.7, rstride=1, cstride=1)
    ax4.set_title('3D Visualization of Frequency Domain', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Frequency X')
    ax4.set_ylabel('Frequency Y')
    ax4.set_zlabel('Magnitude (Log)')
    ax4.view_init(elev=30, azim=45)

    # Denoised image
    ax5 = plt.subplot(gs[1, 1])
    im5 = ax5.imshow(img_back, cmap='gray')
    ax5.set_title('Denoised Image', fontsize=12, fontweight='bold')
    ax5.set_xticks([])
    ax5.set_yticks([])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax)

    # Comparison view with differences
    ax6 = plt.subplot(gs[1, 2])
    difference = np.abs(img_array.astype(np.float32) - img_back.astype(np.float32))
    im6 = ax6.imshow(difference, cmap='hot')
    ax6.set_title('Difference (Noise Removed)', fontsize=12, fontweight='bold')
    ax6.set_xticks([])
    ax6.set_yticks([])
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im6, cax=cax)

    plt.tight_layout()

    # Save the visualization
    plt.savefig(output_path.replace('.jpg', '_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Additional educational information
    print("\n" + "=" * 80)
    print("FOURIER TRANSFORM IMAGE PROCESSING DEMONSTRATION")
    print("=" * 80)
    print(f"\nOriginal image dimensions: {img_array.shape}")
    print(f"Threshold value: {threshold:.2f} (keeping top {threshold_percentage * 100:.1f}% frequencies)")
    print(f"Number of frequencies retained: {np.sum(mask)} out of {mask.size} ({np.sum(mask) / mask.size * 100:.2f}%)")

    # Calculate noise reduction metrics
    noise_power = np.sum(np.abs(img_array - img_back) ** 2)
    signal_power = np.sum(img_back ** 2)
    if signal_power > 0:
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        print(f"Estimated Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    print(f"\nDenoised image saved to: {output_path}")
    print(f"Visualization saved to: {output_path.replace('.jpg', '_visualization.png')}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoise an image using Fourier Transform')
    parser.add_argument('input_image', help='Path to the noisy image')
    parser.add_argument('output_image', help='Path where the denoised image will be saved')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Percentage of frequencies to keep (default: 0.05, lower = more aggressive filtering)')

    args = parser.parse_args()

    fourier_denoise(args.input_image, args.output_image, args.threshold)