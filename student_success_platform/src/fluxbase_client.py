"""
=============================================================================
FLUXBASE CLOUD DATABASE & STORAGE CLIENT
AI-Powered Agentic Smart Campus Intelligence & Student Success Platform
=============================================================================
Implements Fluxbase REST API & AWS S3-backed Storage according to the
official Fluxbase Integration Guide:
  1. SQL Execution Endpoint: POST /api/execute-sql (SELECT, INSERT, UPDATE, DDL)
  2. S3 Storage & Buckets: /api/storage/buckets, /api/storage/upload, /api/storage/url
  3. Support for "smart campus" collaborative project & storage bucket
  4. Environment variable loading from .env (FLUXBASE_PROJECT_ID, FLUXBASE_API_KEY)
=============================================================================
"""

import os
import json
import logging
import requests
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FluxbaseClient")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SSP_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(SSP_DIR)

ENV_CANDIDATES = [
    os.path.join(WORKSPACE_ROOT, ".env"),
    os.path.join(os.path.dirname(WORKSPACE_ROOT), ".env"),
    os.path.join(SSP_DIR, ".env"),
    os.path.join(os.getcwd(), ".env")
]


def load_fluxbase_env():
    """Dynamically loads Fluxbase configuration from .env files."""
    for p in ENV_CANDIDATES:
        if os.path.exists(p):
            load_dotenv(p, override=True)
            if os.getenv("FLUXBASE_API_KEY") or os.getenv("FLUXBASE_PROJECT_ID"):
                break


load_fluxbase_env()


class FluxbaseClient:
    """
    Client for interacting with the Fluxbase Cloud Database & Storage API.
    Connects to the project's PostgreSQL engine and AWS S3 Storage on https://fluxbase.vercel.app.
    """

    DEFAULT_BASE_URL = "https://fluxbase.vercel.app"

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        config_path: Optional[str] = None,
        storage_bucket: Optional[str] = None
    ):
        """
        Initialize the Fluxbase client.
        Prioritizes environment variables from .env over fluxbase_config.json.
        """
        load_fluxbase_env()

        self.api_key = api_key or os.getenv("FLUXBASE_API_KEY")
        self.project_id = project_id or os.getenv("FLUXBASE_PROJECT_ID")
        self.base_url = (os.getenv("FLUXBASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.storage_bucket = storage_bucket or os.getenv("FLUXBASE_STORAGE_BUCKET") or "smart-campus"

        # Fallback to local config file if env vars not provided
        if not self.api_key or not self.project_id:
            if config_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(base_dir, "config", "fluxbase_config.json")

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        if not self.api_key:
                            self.api_key = cfg.get("api_key")
                        if not self.project_id:
                            self.project_id = cfg.get("project_id")
                        if not os.getenv("FLUXBASE_URL") and "base_url" in cfg:
                            self.base_url = cfg["base_url"].rstrip("/")
                except Exception as e:
                    logger.warning(f"Could not load config file {config_path}: {e}")

        if not self.api_key or not self.project_id:
            raise ValueError("Both 'api_key' and 'project_id' are required to connect to Fluxbase.")

        self.execute_url = f"{self.base_url}/api/execute-sql"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # =========================================================================
    # SQL EXECUTION ENDPOINT (POST /api/execute-sql)
    # =========================================================================

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute an arbitrary SQL query (DDL or DML) on the Fluxbase database.
        
        Args:
            sql_query: Valid SQL string.
            
        Returns:
            Dictionary containing 'success', 'rows', 'columns', 'rowCount', 'executionInfo'.
        """
        clean_q = sql_query.strip()
        if not clean_q.startswith("/*"):
            clean_q = f"/* edge_nonce={time.time_ns()} */ " + clean_q
        payload = {
            "projectId": self.project_id,
            "query": clean_q
        }

        try:
            response = requests.post(
                self.execute_url,
                json=payload,
                headers=self.headers,
                timeout=15
            )

            if response.status_code != 200:
                logger.error(f"Fluxbase HTTP Error {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "rows": []
                }

            data = response.json()
            if not data.get("success", False):
                err_data = data.get("error", {})
                err_msg = err_data.get("message") if isinstance(err_data, dict) else str(err_data)
                logger.error(f"Fluxbase SQL Execution Error: {err_msg}")
                return {
                    "success": False,
                    "error": str(err_msg or "Unknown SQL Error"),
                    "rows": []
                }

            result = data.get("result", {})
            return {
                "success": True,
                "rows": result.get("rows", []),
                "columns": result.get("columns", []),
                "rowCount": result.get("rowCount", len(result.get("rows", []))),
                "message": result.get("message", ""),
                "executionInfo": data.get("executionInfo", {})
            }

        except Exception as e:
            logger.error(f"Exception communicating with Fluxbase: {e}")
            return {
                "success": False,
                "error": str(e),
                "rows": []
            }

    # =========================================================================
    # S3 STORAGE & BUCKET MANAGEMENT (/api/storage)
    # =========================================================================

    def list_buckets(self) -> Dict[str, Any]:
        """Lists all storage buckets for the current project."""
        url = f"{self.base_url}/api/storage/buckets?projectId={self.project_id}"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e), "buckets": []}

    def create_bucket(self, name: str = "smart-campus", is_public: bool = False) -> Dict[str, Any]:
        """Creates a storage bucket (name rules: lowercase, alphanumeric + hyphens, 1-63 chars)."""
        url = f"{self.base_url}/api/storage/buckets"
        payload = {"projectId": self.project_id, "name": name, "isPublic": is_public}
        try:
            r = requests.post(url, json=payload, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ensure_smart_campus_bucket(self, bucket_name: Optional[str] = None) -> str:
        """Verifies or auto-creates the 'smart-campus' bucket, returning its ID or name."""
        name = bucket_name or self.storage_bucket or "smart-campus"
        res = self.list_buckets()
        for b in res.get("buckets", []):
            if b.get("name") == name:
                return b.get("id", name)
        c_res = self.create_bucket(name=name)
        if c_res.get("success") and c_res.get("bucket"):
            return c_res["bucket"].get("id", name)
        return name

    def upload_file(
        self,
        file_content: Any,
        filename: str,
        bucket_id_or_name: Optional[str] = None,
        mime_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Uploads a file (PDF, notes, slides, assignment) to Fluxbase S3 Storage.
        Returns: { success: true, file: { id, name, s3_key, size, mime_type } }
        """
        bucket = bucket_id_or_name or self.storage_bucket or "smart-campus"
        url = f"{self.base_url}/api/storage/upload"
        upload_headers = {"Authorization": f"Bearer {self.api_key}"}

        if isinstance(file_content, (bytes, bytearray)):
            files = {"file": (filename, file_content, mime_type)}
        elif hasattr(file_content, "read"):
            files = {"file": (filename, file_content.read(), mime_type)}
        elif isinstance(file_content, str) and os.path.exists(file_content):
            files = {"file": (filename, open(file_content, "rb"), mime_type)}
        else:
            files = {"file": (filename, str(file_content).encode(), mime_type)}

        data = {"bucketId": bucket, "projectId": self.project_id}
        try:
            r = requests.post(url, headers=upload_headers, files=files, data=data, timeout=30)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_presigned_download_url(self, s3_key: str) -> Dict[str, Any]:
        """Generates a 15-minute presigned S3 URL to securely download/render a private file."""
        url = f"{self.base_url}/api/storage/url?s3Key={s3_key}&projectId={self.project_id}"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self, bucket_id_or_name: Optional[str] = None) -> Dict[str, Any]:
        """Lists files stored within a storage bucket."""
        bucket = bucket_id_or_name or self.storage_bucket or "smart-campus"
        url = f"{self.base_url}/api/storage/files?bucketId={bucket}&projectId={self.project_id}"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e), "files": []}

    def delete_file(self, file_id: str, s3_key: str) -> Dict[str, Any]:
        """Deletes a file from Fluxbase S3 Storage."""
        url = f"{self.base_url}/api/storage/files"
        payload = {"fileId": file_id, "s3Key": s3_key, "projectId": self.project_id}
        try:
            r = requests.delete(url, json=payload, headers=self.headers, timeout=10)
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # SCHEMA INITIALIZATION (FACULTY & STUDENTS TABLES)
    # =========================================================================

    def init_schema(self) -> Dict[str, Any]:
        """
        Initialize relational PostgreSQL tables for Faculty, Students, and Advising Logs.
        Executes each DDL statement individually as required by Fluxbase security policy.
        """
        statements = [
            """
            CREATE TABLE IF NOT EXISTS faculty (
                faculty_id VARCHAR(32) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                department VARCHAR(128) NOT NULL,
                designation VARCHAR(128) NOT NULL,
                phone VARCHAR(32),
                specialization VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS students (
                srn VARCHAR(32) PRIMARY KEY,
                student_name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                program VARCHAR(128) DEFAULT 'B.Tech CSE (AI & ML)',
                semester INT DEFAULT 7,
                section VARCHAR(8) DEFAULT 'B',
                risk_level VARCHAR(16) DEFAULT 'LOW',
                dropout_probability FLOAT DEFAULT 0.05,
                assigned_advisor_id VARCHAR(32) REFERENCES faculty(faculty_id),
                approved_units_sem1 INT DEFAULT 5,
                approved_units_sem2 INT DEFAULT 5,
                tuition_fees_up_to_date INT DEFAULT 1,
                scholarship_holder INT DEFAULT 0,
                debtor INT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS academic_advising_logs (
                log_id SERIAL PRIMARY KEY,
                srn VARCHAR(32) REFERENCES students(srn) ON DELETE CASCADE,
                faculty_id VARCHAR(32) REFERENCES faculty(faculty_id),
                intervention_type VARCHAR(64) NOT NULL,
                notes TEXT NOT NULL,
                action_prescribed TEXT,
                status VARCHAR(32) DEFAULT 'Open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]

        results = []
        for stmt in statements:
            res = self.execute_query(stmt)
            results.append(res)
            if not res.get("success", False):
                logger.error(f"Failed to execute schema statement: {res.get('error')}")
                return res

        return {"success": True, "details": results}

    def seed_academic_data(self) -> Dict[str, Any]:
        """Seeds initial academic cohort into Fluxbase."""
        faculties = [
            ("FAC_AIML_01", "Dr. Ramesh Kumar", "ramesh.kumar@reva.edu.in", "AI & ML", "Associate Professor", "+91-9876543210", "Deep Learning & Neural Networks"),
            ("FAC_AIML_02", "Prof. Ananya Sharma", "ananya.s@reva.edu.in", "AI & ML", "Assistant Professor", "+91-9876543211", "Computer Vision & Pattern Recognition"),
            ("FAC_AIML_03", "Dr. Rajesh V", "rajesh.v@reva.edu.in", "AI & ML", "Professor & Head", "+91-9876543212", "Reinforcement Learning & Robotics")
        ]

        students = [
            ("R24EA807", "Sumith U", "sumith.u@reva.edu.in", "LOW", 0.042, "FAC_AIML_01", 6, 6, 1, 1, 0),
            ("R24EA818", "Rohan Verma", "rohan.v@reva.edu.in", "HIGH", 0.785, "FAC_AIML_01", 1, 0, 0, 0, 1),
            ("R24EA842", "Priya Nair", "priya.n@reva.edu.in", "MEDIUM", 0.362, "FAC_AIML_02", 4, 3, 1, 0, 0),
            ("R24EA801", "Aarav Sharma", "aarav.s@reva.edu.in", "LOW", 0.089, "FAC_AIML_03", 5, 5, 1, 0, 0),
            ("R24EA895", "Sneha Patel", "sneha.p@reva.edu.in", "HIGH", 0.812, "FAC_AIML_01", 2, 1, 0, 0, 1)
        ]

        results = []
        for fac in faculties:
            q = f"""
            INSERT INTO faculty (faculty_id, name, email, department, designation, phone, specialization)
            VALUES ('{fac[0]}', '{fac[1]}', '{fac[2]}', '{fac[3]}', '{fac[4]}', '{fac[5]}', '{fac[6]}')
            ON CONFLICT (faculty_id) DO UPDATE SET
                name = EXCLUDED.name, email = EXCLUDED.email, designation = EXCLUDED.designation;
            """
            results.append(self.execute_query(q))

        for stu in students:
            q = f"""
            INSERT INTO students (srn, student_name, email, risk_level, dropout_probability, assigned_advisor_id,
                                  approved_units_sem1, approved_units_sem2, tuition_fees_up_to_date, scholarship_holder, debtor)
            VALUES ('{stu[0]}', '{stu[1]}', '{stu[2]}', '{stu[3]}', {stu[4]}, '{stu[5]}',
                    {stu[6]}, {stu[7]}, {stu[8]}, {stu[9]}, {stu[10]})
            ON CONFLICT (srn) DO UPDATE SET
                risk_level = EXCLUDED.risk_level,
                dropout_probability = EXCLUDED.dropout_probability,
                assigned_advisor_id = EXCLUDED.assigned_advisor_id;
            """
            results.append(self.execute_query(q))

        return {"success": True, "details": results}

    # =========================================================================
    # HIGH-LEVEL DATA ACCESS METHODS
    # =========================================================================

    def get_students(self, risk_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve students from Fluxbase, optionally filtered by risk level."""
        if risk_filter:
            sql = f"""
            SELECT s.*, f.name as advisor_name 
            FROM students s
            LEFT JOIN faculty f ON s.assigned_advisor_id = f.faculty_id
            WHERE s.risk_level = '{risk_filter.upper()}'
            ORDER BY s.dropout_probability DESC
            LIMIT {limit};
            """
        else:
            sql = f"""
            SELECT s.*, f.name as advisor_name 
            FROM students s
            LEFT JOIN faculty f ON s.assigned_advisor_id = f.faculty_id
            ORDER BY s.dropout_probability DESC
            LIMIT {limit};
            """
        res = self.execute_query(sql)
        return res.get("rows", [])

    def get_student_by_srn(self, srn: str) -> Optional[Dict[str, Any]]:
        """Retrieve single student record by institutional SRN."""
        alias_map = {
            "STU-807": "R24EA807",
            "STU-818": "R24EA818",
            "STU-842": "R24EA842",
            "STU-801": "R24EA801",
            "STU-895": "R24EA895"
        }
        resolved = alias_map.get(srn, srn)
        sql = f"""
        SELECT s.*, f.name as advisor_name, f.email as advisor_email
        FROM students s
        LEFT JOIN faculty f ON s.assigned_advisor_id = f.faculty_id
        WHERE s.srn = '{resolved}';
        """
        res = self.execute_query(sql)
        rows = res.get("rows", [])
        return rows[0] if rows else None

    def get_faculty_list(self) -> List[Dict[str, Any]]:
        """Retrieve all registered faculty members and academic mentors."""
        sql = """
        SELECT f.*, COUNT(s.srn) as assigned_mentees_count
        FROM faculty f
        LEFT JOIN students s ON f.faculty_id = s.assigned_advisor_id
        GROUP BY f.faculty_id, f.name, f.email, f.department, f.designation, f.phone, f.specialization, f.created_at
        ORDER BY f.faculty_id ASC;
        """
        res = self.execute_query(sql)
        return res.get("rows", [])

    def log_faculty_intervention(
        self,
        srn: str,
        faculty_id: str,
        intervention_type: str,
        notes: str,
        status: str = "Open"
    ) -> Dict[str, Any]:
        """Record an official faculty advising or remedial intervention note."""
        clean_notes = notes.replace("'", "''")
        sql = f"""
        INSERT INTO academic_advising_logs (srn, faculty_id, intervention_type, notes, status)
        VALUES ('{srn}', '{faculty_id}', '{intervention_type}', '{clean_notes}', '{status}')
        RETURNING log_id, created_at;
        """
        return self.execute_query(sql)

    def get_advising_history(self, srn: str) -> List[Dict[str, Any]]:
        """Retrieve all advising history and counseling notes for a student."""
        sql = f"""
        SELECT l.*, f.name as faculty_name, f.designation as faculty_designation
        FROM academic_advising_logs l
        LEFT JOIN faculty f ON l.faculty_id = f.faculty_id
        WHERE l.srn = '{srn}'
        ORDER BY l.created_at DESC;
        """
        res = self.execute_query(sql)
        return res.get("rows", [])


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  FLUXBASE INTEGRATION VERIFICATION (DATABASE & STORAGE)")
    print("=" * 75)

    client = FluxbaseClient()
    print(f"1. Connected Project ID: {client.project_id}")
    print(f"2. Base URL: {client.base_url}")
    print(f"3. Storage Bucket: {client.storage_bucket}")

    buckets = client.list_buckets()
    print(f"4. Active Buckets in Project: {[b.get('name') for b in buckets.get('buckets', [])]}")

    test_q = client.execute_query("SELECT COUNT(*) as student_count FROM students;")
    print(f"5. Test SQL Query: {test_q.get('rows')}")
    print("=" * 75 + "\n")
