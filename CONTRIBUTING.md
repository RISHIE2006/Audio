# Contributing to Audio Verifier

First off, thank you for considering contributing to Audio Verifier! It's people like you that make this tool better for everyone.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Install dependencies** as outlined in the `README.md`.
4. **Create a new branch** for your feature or bug fix.

## Development Setup

- Ensure you have Node.js (v18+) and Python (v3.8+) installed.
- Run `npm install` to install frontend and server dependencies.
- The Python virtual environment is created automatically on the first start, or you can manually create it and run `pip install -r requirements.txt`.
- Start the development environment with `npm run dev`.

## Making Changes

- Ensure your code follows the existing style (TypeScript for frontend/backend, Python for machine learning scripts).
- If you're modifying the UI, check that it still looks good and remains responsive.
- If you're touching the ML pipeline, ensure you test model training with `python train_model.py`.

## Submitting a Pull Request

1. Commit your changes with a clear commit message.
2. Push your branch to your fork.
3. Open a Pull Request against the `main` branch.
4. Describe your changes in detail in the PR description.

## Reporting Bugs

If you find a bug, please create an issue with the following details:
- Steps to reproduce the bug.
- Expected behavior vs actual behavior.
- Any relevant logs or error messages.

Thank you for contributing!
