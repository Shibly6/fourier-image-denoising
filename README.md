# 🖼️ Image Denoising using 2D Fourier Transform

An educational Python implementation demonstrating image denoising through frequency domain filtering using the **2D Fast Fourier Transform (FFT)**. 

> 🎉 **Note:** This project was created just for fun as a practical application exploration for the **Mathematics II (Spring 2025)** course.

An explanation of how it works: it isolates and retains significant frequencies while filtering out low-amplitude high-frequency noise, outputting both the cleaned image and a comprehensive 2D/3D frequency spectrum visualization dashboard.

## ✨ Features
* **Frequency Domain Filtering:** Converts grayscale images into frequency components using `np.fft.fft2`.
* **Dynamic Thresholding:** Retains only the top percentile of significant frequencies based on user configuration.
* **3D Spectrum Plotting:** Renders a gorgeous 3D visualization of the log-scaled frequency domain.
* **Performance Analysis:** Automatically calculates the estimated Signal-to-Noise Ratio (SNR) in dB post-processing.
* **Robust Visualization:** Utilizes `matplotlib.gridspec` to display original vs. filtered spectra, results, and removed noise maps side-by-side.
