# Game Project

A small AI test game built with Python and Pygame, leveraging machine learning models from the Transformers library.

## Features

- ...

---

## Getting Started

### Prerequisites

- Python 3.12+
- Pip package manager

### Starting Python on Windows 11

1. Download and install [Python for Windows](https://www.python.org/downloads/windows/).
   During installation, ensure that **Add Python to PATH** is checked.
2. Open the **Start** menu, search for **Command Prompt** or **PowerShell**, and launch it.
3. Verify the installation:

   ```powershell
   python --version
   ```

4. Start the interactive interpreter or run a script:

   ```powershell
   python
   python your_script.py
   ```

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository-url.git
   cd game-project
   ```

2. Create and activate a virtual environment:

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows**

   ```powershell
   python -m venv .venv
   # PowerShell
   .\.venv\Scripts\Activate
   # Command Prompt
   .venv\Scripts\activate.bat
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Upgrade pip:

   ```bash
   pip install --upgrade pip
   ```

---

## Running the Game

To start the game, execute the main script:

```bash
python src/main.py
```

---

## Testing

### Running Tests

Run all tests with code coverage:

```bash
pytest --cov=src --cov-report=html tests/
```

### Viewing Coverage Report

After running the tests, you can find the coverage report in the `htmlcov` directory. Open the `index.html` file in a browser to see detailed coverage results.

---

## Managing the Environment

### Activating the Virtual Environment

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
# PowerShell
.\.venv\Scripts\Activate
# Command Prompt
.\venv\Scripts\activate.bat
```

### Deactivating the Virtual Environment

```bash
deactivate
```

---

## Contributing

...

---

## License

...

---

## Author

...
