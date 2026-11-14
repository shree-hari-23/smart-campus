# 🎓 Smart Campus Intelligence & Student Success Platform (PACD-RAG)

An agentic, AI-powered academic intelligence and student success ecosystem built for **REVA University**. The platform provides early academic warning indicators, automated concept remediation, AI tutoring via **Google Gemini**, and faculty automation tools.

---

## 🌟 Key Features

### 1. 🤖 Multi-Model ML Risk Prediction Engine
* **4-Model Ensemble:** Combines XGBoost, Random Forest, and dual CatBoost models.
* **Explainable AI (SHAP):** Deconstructs individual student risk scores into exact feature contributions (attendance, quiz trends, submission recency).
* **3-Way Trajectory Predictor:** Forecasts Best Case, Expected, and Worst Case academic trajectories.
* **Trained on Benchmark Data:** Built on UCI Machine Learning Repository Dataset #697 (4,424 records).

### 2. ⚡ Google Gemini AI Gateway
* Powered by Google's **Gemini 2.5 Flash** model.
* **OmniAssistant Chatbot:** ReAct-style agent capable of intent classification, course RAG querying, quiz generation, and prerequisite explanation.
* **SSE Streaming:** Real-time token streaming with live fallback safety mechanisms.

### 3. 🕸️ Prerequisite Knowledge Graph
* Directed Acyclic Graph (DAG) powered by **NetworkX** with 68 concepts and 112 prerequisite dependency edges.
* Pinpoints root-cause conceptual gaps (e.g., identifies Linear Algebra / Calculus weaknesses when struggling with Backpropagation).

### 4. 📚 Prerequisite-Aware RAG (ChromaDB)
* Dual RAG engines (**Subject RAG** & **Weak-Topic RAG**) indexing course PDFs, lecture presentations, and faculty notes.
* Context-augmented query generation using semantic similarity.

### 5. 🗄️ Fluxbase Cloud Database & Collaborative Storage
* Cloud-hosted PostgreSQL schema (23 relational tables) managing authentication, course enrollments, learning materials, and AI logs.
* Direct integration with Fluxbase collaborative storage buckets.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | FastAPI, Uvicorn, Starlette, Pydantic |
| **AI / LLM** | Google Gemini (`google-generativeai`, `google-genai`) |
| **Machine Learning** | XGBoost, CatBoost, Scikit-Learn, SHAP, NumPy, Pandas |
| **Vector DB** | ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Knowledge Graph** | NetworkX (DAG analysis) |
| **Database** | Fluxbase Cloud (PostgreSQL) + REST Client |
| **Document Processing**| PyPDF, Python-PPTX, Python-Docx, ReportLab |
| **Security & Auth** | JWT (`python-jose`), Bcrypt, Passlib |
| **Frontend** | Responsive Vanilla HTML5, CSS3, JavaScript (Fetch + SSE) |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+ (Recommended Python 3.11 or 3.12)
* Node.js (for SMTP mail notifications)
* Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shree-hari-23/smart-campus.git
   cd smart-campus
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   npm install nodemailer dotenv
   ```

4. **Configure Environment Variables:**
   Copy the example environment file and add your credentials:
   ```bash
   cp .env.example .env
   ```
   Set your keys in `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   FLUXBASE_PROJECT_ID=your_fluxbase_project_id
   FLUXBASE_API_KEY=your_fluxbase_api_key
   FLUXBASE_BUCKET_ID=smart-campus
   ```

5. **Start the Platform:**
   ```bash
   python app.py
   ```

---

## 🌐 Port Access

Once running, the application is accessible on:
* **Primary Web Portal:** `http://localhost:8000/`
* **Mirror Portal:** `http://localhost:8002/`
* **Interactive API Documentation:** `http://localhost:8000/docs`

---

## 🔑 Test Credentials

| Role | Email | Password | Identifier |
| :--- | :--- | :--- | :--- |
| 🎓 **Student** | `sumithu.dev@gmail.com` | `password123` | SRN: `R24EA807` |
| 👩‍🏫 **Faculty** | `ramesh.kumar@reva.edu.in` | `password123` | Faculty ID: `FAC_AIML_01` |
| 🔐 **Admin** | `admin@reva.edu.in` | `admin123` | Role: `ADMIN` |

---

## 📂 Project Structure

```
smart-campus/
├── app.py                             # Dual-port FastAPI launcher (8000 / 8002)
├── index.html                         # Login portal
├── student.html                       # Student analytics & OmniAssistant dashboard
├── faculty.html                       # Faculty cohort console & assignment manager
├── requirements.txt                   # Complete Python dependencies
├── .env.example                       # Template for environment variables
├── docs/                              # Comprehensive documentation & viva guides
│   ├── PACD_RAG_Project_Explanation.pdf
│   ├── ML_Models_QnA_Viva_Prep.pdf
│   └── fluxbase-integration-guide.pdf
├── models/                            # Trained ML model artifacts
│   ├── model_1_early_warning_xgb.pkl  # XGBoost early warning classifier
│   ├── model_2_engagement_rf.pkl      # Random Forest engagement model
│   ├── model_3_mastery_catboost.cbm   # CatBoost concept mastery model
│   ├── model_4_champion_catboost.cbm  # CatBoost champion predictor
│   └── ensemble_metadata.pkl          # Ensemble weights & thresholds
├── student_success_platform/
│   ├── src/
│   │   ├── api_server.py              # Core FastAPI REST & SSE routers
│   │   ├── gemini_ai_service.py       # Google Gemini AI Client & gateway
│   │   ├── omni_assistant_engine.py   # ReAct conversational agent
│   │   ├── prerequisite_graph.py      # NetworkX dependency graph traversal
│   │   ├── subject_rag_engine.py      # ChromaDB subject retrieval engine
│   │   ├── weak_topic_rag_engine.py   # Remediation material retrieval
│   │   ├── academic_recovery_predictor.py # 3-way trajectory predictor
│   │   └── fluxbase_client.py         # Cloud database & bucket interface
│   └── data/
│       └── concept_prerequisites.json # 68 concepts & 112 prerequisite relationships
├── js/                                # Frontend application scripts
└── css/                               # Portal stylesheets
```

---

## 📖 Documentation
Detailed technical documentation and presentation guides are available in the [`docs/`](docs/) directory:
* **[PACD RAG Full Project Explanation](docs/PACD_RAG_Project_Explanation.pdf)**
* **[Machine Learning Viva Q&A Guide](docs/ML_Models_QnA_Viva_Prep.pdf)**
* **[Fluxbase Integration Guide](docs/fluxbase-integration-guide.pdf)**

---

## 📄 License
This project is developed for educational and academic research purposes at REVA University.
