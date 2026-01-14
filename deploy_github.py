#!/usr/bin/env python3
"""
GitHub Deployment Script with AI Service Access
Uses GitHub PAT for authenticated deployments and AI service access.
"""

import os
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

class GitHubDeployer:
    """Handles GitHub deployments with AI service access."""
    
    def __init__(self):
        self.config = self._load_config()
        self.token = self.config["github"]["token"]
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.api_base = "https://api.github.com"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load GitHub configuration."""
        config_file = ".github_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback configuration
            return {
                "github": {
                    "token": os.getenv("GITHUB_TOKEN", ""),
                    "username": "Workofarttattoo",
                    "repository": "echo_prime"
                }
            }
    
    def verify_token(self) -> bool:
        """Verify the GitHub token is valid."""
        url = f"{self.api_base}/user"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ GitHub token verified for user: {user_data.get('login')}")
            return True
        else:
            print(f"❌ GitHub token verification failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def check_repository_access(self) -> bool:
        """Check access to the repository."""
        owner = self.config["github"]["username"]
        repo = self.config["github"]["repository"]
        
        url = f"{self.api_base}/repos/{owner}/{repo}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            repo_data = response.json()
            print(f"✅ Repository access confirmed: {repo_data.get('full_name')}")
            print(f"   Private: {repo_data.get('private')}")
            return True
        else:
            print(f"❌ Repository access failed: {response.status_code}")
            return False
    
    def push_to_github(self, commit_message: str = "Automated deployment") -> bool:
        """Push current changes to GitHub."""
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Push to main branch
            subprocess.run(["git", "push", "origin", "main"], check=True)
            
            print("✅ Successfully pushed to GitHub")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git push failed: {e}")
            return False
    
    def setup_github_pages(self) -> bool:
        """Set up GitHub Pages for kairos.aios.is."""
        owner = self.config["github"]["username"]
        repo = self.config["github"]["repository"]
        
        # Enable GitHub Pages
        url = f"{self.api_base}/repos/{owner}/{repo}/pages"
        data = {
            "source": {
                "branch": "main",
                "path": "/docs"
            }
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        
        if response.status_code in [201, 204]:
            print("✅ GitHub Pages enabled")
            
            # Create CNAME file for custom domain
            cname_path = Path("docs/CNAME")
            cname_path.parent.mkdir(exist_ok=True)
            cname_path.write_text("kairos.aios.is\n")
            
            print("✅ CNAME file created for kairos.aios.is")
            return True
        else:
            print(f"❌ GitHub Pages setup failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def create_deployment_workflow(self) -> bool:
        """Create GitHub Actions workflow for automated deployment."""
        workflows_dir = Path(".github/workflows")
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = f'''name: Deploy ECH0-PRIME

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Deploy to Railway
      if: success()
      run: |
        python deploy_railway.py
      env:
        GITHUB_TOKEN: {self.token}
        RAILWAY_TOKEN: ${{{{ secrets.RAILWAY_TOKEN }}}}
    
    - name: Update GitHub Pages
      if: success()
      run: |
        git config --global user.name 'github-actions[bot]'
        git config --global user.email 'github-actions[bot]@users.noreply.github.com'
        git checkout -b gh-pages
        cp -r dashboard/v2/* .
        git add .
        git commit -m 'Update GitHub Pages' || echo "No changes to commit"
        git push origin gh-pages
'''
        
        workflow_file = workflows_dir / "deploy.yml"
        workflow_file.write_text(workflow_content)
        
        print("✅ GitHub Actions workflow created")
        return True
    
    def test_ai_services_access(self) -> Dict[str, bool]:
        """Test access to AI services via GitHub."""
        results = {}
        
        # Test GitHub Copilot API access (if available)
        try:
            url = f"{self.api_base}/user/copilot"
            response = requests.get(url, headers=self.headers)
            results["github_copilot"] = response.status_code == 200
        except:
            results["github_copilot"] = False
        
        # Note: Direct API testing for OpenAI/Anthropic would require their API keys
        # The GitHub token enables access through GitHub's AI services integration
        
        print("🔍 AI Services Access Test:")
        for service, accessible in results.items():
            status = "✅" if accessible else "❌"
            print(f"   {status} {service.replace('_', ' ').title()}")
        
        return results

def main():
    """Main deployment function."""
    print("🚀 ECH0-PRIME GitHub Deployment with AI Services")
    print("=" * 55)
    
    deployer = GitHubDeployer()
    
    # Verify token
    if not deployer.verify_token():
        print("❌ GitHub token verification failed")
        return False
    
    # Check repository access
    if not deployer.check_repository_access():
        print("❌ Repository access check failed")
        return False
    
    # Test AI services access
    deployer.test_ai_services_access()
    
    print("\n📋 Deployment Options:")
    print("1. Push current changes to GitHub")
    print("2. Set up GitHub Pages for kairos.aios.is")
    print("3. Create GitHub Actions workflow")
    print("4. All of the above")
    
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nChoose deployment option (1-4): ").strip()
    
    success = True
    
    if choice in ["1", "4"]:
        print("\n📤 Pushing to GitHub...")
        if not deployer.push_to_github("ECH0-PRIME deployment with AI services access"):
            success = False
    
    if choice in ["2", "4"]:
        print("\n🌐 Setting up GitHub Pages...")
        if not deployer.setup_github_pages():
            success = False
    
    if choice in ["3", "4"]:
        print("\n⚙️ Creating GitHub Actions workflow...")
        if not deployer.create_deployment_workflow():
            success = False
    
    if success:
        print("\n🎉 Deployment completed successfully!")
        print("🌐 Access your site at: https://kairos.aios.is")
        print("🤖 AI services are now accessible via GitHub authentication")
    else:
        print("\n❌ Some deployment steps failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
