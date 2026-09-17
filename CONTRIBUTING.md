# Contributing to Indentation

First off, thank you for your interest in contributing! This project is part of an academic research effort. Whether you are reporting a bug, suggesting a feature, or writing code, your help is welcome.

## Reporting Bugs or Issues
If you encounter a bug, a mathematical issue, or have a question about the implementation, you can either:
1. **Open a GitHub Issue:** Please include a clear description of the problem, the exact model and probe type you were using, and a minimal code snippet or CSV data that triggered the error.
2. **Email the Developer:** You can directly email the lead developer, Hasan Seraj (hasan.seraj25@gmail.com), with your questions or bug reports.

## Suggesting Enhancements and Roadmap
We have an active roadmap for expanding this package into a comprehensive tool for soft tissue mechanics and biomaterials. If you have a specific mathematical model or feature you would like to see, please open an Issue and tag it as an "Enhancement". 

Our current future plans include:
* **New Material Models:** Adding hyperelastic, visco-hyperelastic, and poroelastic constitutive models.
* **File I/O:** Native support for reading proprietary AFM nanoindentation files (e.g., JPK files).
* **Preprocessing:** Automated contact-point detection, baseline correction, and thermal drift removal.
* **Experimental Metrics:** Converting time-domain data directly to Dynamic Mechanical Analysis (DMA) parameters.

If you are an expert in any of these areas and would like to contribute code or equations, please reach out!

## Pull Requests
If you have written a fix or a new feature and want to submit it:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m "Add some feature"`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request.

Please ensure your code passes the existing `pytest tests/` suite before submitting.