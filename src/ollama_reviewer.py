import os
import requests
from typing import List, Dict


class DeepSeekReviewer:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.model = "deepseek-r1:8b"

    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096
            },
            "stream": False  # Set to True if you want streaming responses
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def review_code(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding="utf-8") as f:
            code = f.read()

        prompt = f"""Perform a comprehensive code review for this Python file:
{code}

Analyze the following aspects:
1. Potential bugs and errors
2. Code quality issues
3. Performance optimizations
4. Security vulnerabilities
5. PEP-8 compliance

Format your response in Markdown with section headers."""

        return {
            "file": file_path,
            "review": self._generate(prompt)
        }

    def review_project(self, project_path: str) -> List[Dict]:
        reviews = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    reviews.append(self.review_code(file_path))
        return reviews


if __name__ == "__main__":
    reviewer = DeepSeekReviewer()
    project_path = r"C:\pythonproject\forex_robot"
    results = reviewer.review_project(project_path)

    for review in results:
        print(f"\n{'=' * 40}\nReview for {review['file']}\n{'=' * 40}")
        print(review['review'])
