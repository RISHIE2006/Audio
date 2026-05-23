import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import { spawn } from "child_process";
import { platform } from "os";

function getPipPath(): string {
  const venvPath = path.join(process.cwd(), "venv");
  const isWindows = platform() === "win32";
  return isWindows
    ? path.join(venvPath, "Scripts", "pip.exe")
    : path.join(venvPath, "bin", "pip");
}

async function startServer() {
  try {
    if (!fs.existsSync(path.join(process.cwd(), "venv"))) {
      console.log("Creating Python virtual environment...");
      spawn("python", ["-m", "venv", "venv"]).on("close", (code) => {
        if (code === 0) {
          console.log("Installing Python dependencies...");
          const pipProcess = spawn(getPipPath(), ["install", "-r", "requirements.txt"], { stdio: "inherit" });
          pipProcess.on("error", (err) => {
            console.error("Warning: Failed to run pip install:", err);
          });
        }
      });
    } else {
      console.log("Python venv exists. Installing dependencies...");
      const pipProcess = spawn(getPipPath(), ["install", "-r", "requirements.txt"], { stdio: "inherit" });
      pipProcess.on("error", (err) => {
        console.error("Warning: Failed to run pip install:", err);
      });
    }
  } catch (err) {
    console.error("Warning: Failed to setup Python environment.", err);
  }

  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  const upload = multer({ dest: "uploads/" });

  // API Route for prediction
  app.post("/api/predict", upload.single("audio"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = path.resolve(req.file.path);
    const fileName = req.file.originalname;

    // Use Python script for prediction
    // We'll call a wrapper script that uses the user's dsp_logic and the trained model
    const isWindows = platform() === "win32";
    const pythonPath = isWindows
      ? path.join(process.cwd(), "venv", "Scripts", "python.exe")
      : path.join(process.cwd(), "venv", "bin", "python3");
    const pythonCmd = fs.existsSync(pythonPath) ? pythonPath : "python";
    const pythonProcess = spawn(pythonCmd, ["predict_wrapper.py", filePath], { cwd: process.cwd() });

    let dataString = "";
    let errorString = "";

    pythonProcess.stdout.on("data", (data: Buffer) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on("data", (data: Buffer) => {
      errorString += data.toString();
      console.error(`Python stderr: ${data.toString()}`);
    });

    pythonProcess.on("close", (code: number) => {
      // Clean up uploaded file
      try {
        fs.unlinkSync(filePath);
      } catch (err) {
        console.error("Error deleting temp file:", err);
      }

      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        return res.status(500).json({ error: `Prediction failed: ${errorString || "Unknown error"}` });
      }

      try {
        const result = JSON.parse(dataString);
        res.json({ filename: fileName, ...result });
      } catch (e) {
        console.error(`Failed to parse: ${dataString}`);
        res.status(500).json({ error: `Failed to parse prediction result: ${e}` });
      }
    });
  });

  app.post("/api/train", async (req, res) => {
    const isWindows = platform() === "win32";
    const pythonPath = isWindows
      ? path.join(process.cwd(), "venv", "Scripts", "python.exe")
      : path.join(process.cwd(), "venv", "bin", "python3");
    const pythonCmd = fs.existsSync(pythonPath) ? pythonPath : "python";

    const trainProcess = spawn(pythonCmd, ["train_model.py", "--force"], { cwd: process.cwd() });
    let outputString = "";
    let errorString = "";

    trainProcess.stdout.on("data", (data: Buffer) => {
      outputString += data.toString();
    });

    trainProcess.stderr.on("data", (data: Buffer) => {
      errorString += data.toString();
      console.error(`Train stderr: ${data.toString()}`);
    });

    trainProcess.on("close", (code: number) => {
      if (code !== 0) {
        console.error(`Train process exited with code ${code}`);
        return res.status(500).json({ error: `Training failed: ${errorString || "Unknown error"}` });
      }
      res.json({ message: "Model retrained successfully.", output: outputString.trim() });
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
