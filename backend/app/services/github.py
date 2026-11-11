import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.core.config import settings


def _github_get(url: str) -> object:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ProfilePilot-AI",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 404:
            return None
        if error.code == 403:
            raise ValueError("GitHub rate limit reached. Please try again later.") from error
        raise ValueError("GitHub could not complete the request.") from error
    except URLError as error:
        raise ValueError("Could not connect to GitHub.") from error


def import_repositories(username: str) -> list[dict]:
    username = username.strip()
    if not username or len(username) > 39:
        raise ValueError("Enter a valid public GitHub username.")
    repositories = _github_get(
        f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    )
    if not isinstance(repositories, list):
        raise ValueError("GitHub username was not found.")

    selected = [
        repo for repo in repositories
        if isinstance(repo, dict) and not repo.get("fork")
    ][:settings.max_github_repositories]
    imported: list[dict] = []
    for repo in selected:
        full_name = str(repo.get("full_name", ""))
        readme = _github_get(
            f"https://api.github.com/repos/{full_name}/readme"
        )
        readme_text = ""
        if isinstance(readme, dict) and readme.get("content"):
            import base64
            try:
                readme_text = base64.b64decode(readme["content"]).decode(
                    "utf-8", errors="replace"
                )[:settings.max_readme_size]
            except (ValueError, TypeError):
                readme_text = ""
        imported.append({
            "name": repo.get("name", full_name),
            "description": repo.get("description") or "",
            "language": repo.get("language") or "Not specified",
            "url": repo.get("html_url", ""),
            "topics": repo.get("topics") or [],
            "readme": readme_text,
            "stars": repo.get("stargazers_count", 0),
        })
    return imported


def repository_chunks(repository: dict) -> list[dict]:
    text = (
        f"GitHub repository: {repository['name']}\n"
        f"Description: {repository['description']}\n"
        f"Primary language: {repository['language']}\n"
        f"Topics: {', '.join(repository['topics']) or 'None'}\n"
        f"URL: {repository['url']}\n"
        f"README:\n{repository['readme'] or 'No README available.'}"
    )
    chunks = []
    for index in range(0, len(text), 1000):
        chunks.append({
            "id": f"github-{repository['name']}-c{index // 1000 + 1:03d}",
            "text": text[index:index + 1000],
            "filename": repository["name"],
            "page": None,
            "repository_name": repository["name"],
            "repository_url": repository["url"],
        })
    return chunks
