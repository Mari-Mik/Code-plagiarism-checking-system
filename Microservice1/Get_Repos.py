import requests
import subprocess
import os

# GitHub API endpoint for searching repositories
GITHUB_API_URL = "https://api.github.com/search/repositories"

# Function to fetch repositories based on a search query
def fetch_repositories(query, max_results=3, language=None, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    params = {
        "q": query,  # search query (e.g., language, topic, etc.)
        "per_page": max_results,  # limit the number of results
    }
    
    if language:
        params["q"] += f" language:{language}"

    response = requests.get(GITHUB_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()['items']
    else:
        print(f"Error fetching repositories: {response.status_code}")
        return []

# Function to clone repositories
def clone_repositories(repositories, clone_directory="cloned_repos"):
    if not os.path.exists(clone_directory):
        os.makedirs(clone_directory)
    
    for repo in repositories:
        repo_url = repo['html_url'] + ".git"  # Clone URL is the html_url with '.git'
        print(f"Cloning repository: {repo['name']} from {repo_url}")
        
        # Change to the directory where you want to clone
        subprocess.run(["git", "clone", repo_url], cwd=clone_directory)

# Main function to test the script
def main():
    token = "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"  # Replace with your GitHub personal access token
    query = "machine learning"  # You can change this to any keyword, e.g., 'python', 'data science', etc.
   # language = "Python"  # Optional, specify programming language like 'Python', 'JavaScript', etc.
    
    repositories = fetch_repositories(query, max_results=3,  token=token) #language=language,
    
    # Clone the repositories to the local environment
    if repositories:
        clone_repositories(repositories)
    else:
        print("No repositories found to clone.")

if __name__ == "__main__":
    main()
