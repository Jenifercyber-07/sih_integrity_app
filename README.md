# 🛡️ CV Pipeline Integrity Assurance System
**Smart India Hackathon (SIH) Prototype**
*Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines*

---

## 📌 Architecture Overview

In distributed machine learning and multi-contributor CV pipelines, adversaries or compromised nodes can inject malicious training data, poison model weights via backdoor attacks, or tamper with inference predictions. This system establishes an end-to-end cryptographic and explainability trust protocol across three core layers:

```
[ Contributor Node ] 
        │
        ▼ (Upload Asset)
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: Data Integrity Assurance                            │
│ • Byte-level SHA-256 Hash Chaining                           │
│ • Detects Man-in-the-Middle (MitM) & asset corruption        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 2: Model Structural Fingerprinting                     │
│ • Cryptographic checksum of layer tensor weights & shapes    │
│ • Validates against Golden Baseline Fingerprint              │
│ • Flags unauthorized fine-tuning & weight poisoning          │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 3: Inference Integrity & Explainable AI (XAI)          │
│ • Grad-CAM Saliency Localization Map Generation              │
│ • Visually confirms model focused on genuine features        │
│ • Rules out adversarial shortcut trigger patches             │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 4: Cryptographic Manifest & Audit Ledger               │
│ • Deterministic JSON-LD Manifest Sealed with SHA-256 HMAC    │
│ • Appended to Immutable Block Audit Trail                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Navigate to Project
```bash
cd sih_integrity_app
```

### 2. Dependencies
Dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Note: To install CPU-optimized PyTorch quickly on Windows, use `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

### 3. Launch the Server
```bash
python app.py
```

### 4. Access the Dashboard
Open your browser at:
👉 **`http://127.0.0.1:5000`**

---

## 🔬 How to Demo to SIH Judges

The web application includes an interactive **Live Judge Demonstration Switchboard** so you can demonstrate real-time attack detection on stage:

1. **Happy Path (Normal Operation)**:
   - Select an authorized node (`Node_01 (Certified AI Lab)`).
   - Upload any test image (`sample_car.jpg` is provided in the project folder).
   - Click **"Process & Cryptographically Audit"**.
   - Show the judges:
     - All 3 Layer status indicators turn **Green (`Verified Authentic`, `Weights Intact`, `Grad-CAM Grounded`)**.
     - Side-by-side view shows the original input and the Explainable AI (Grad-CAM) saliency heat map.
     - A digitally signed cryptographic manifest is created and recorded into the Immutable Audit Ledger.

2. **Attack Demo 1: In-Transit Data Tampering**:
   - Check the **"Simulate In-Transit Data Tamper"** box.
   - Click audit.
   - Layer 1 turns **Red (`TAMPER DETECTED`)**, showing that byte hashes do not match the expected signature.

3. **Attack Demo 2: Model Weight Poisoning (Backdoor Injection)**:
   - Click **"Poison Model"** on the switchboard.
   - Click audit.
   - Layer 2 immediately flags **`❌ MODEL POISONED`**, proving the runtime weights checksum deviates from the **Golden Model Baseline Fingerprint**.
   - Click **"Poison Model"** again to reset back to baseline.

4. **Immutable Ledger Block Explorer**:
   - Scroll down to the Ledger table to show every audit log sealed with a unique cryptographic signature, timestamp, and contributor identifier.
