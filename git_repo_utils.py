import glob
import os

from git import InvalidGitRepositoryError, Repo

from code_text_splitter import LanguageExtension

# Provide the repo_path to start searching for Git repositories
base_path = "./source_documents"

# Specify the accepted file extensions in a list
extensions = [e.value for e in LanguageExtension]

class GitRepoUtils:

    def __init__(
        self,
        source_path: str = None,
    ):
        self.source_path = source_path
        if self.source_path is None or len(self.source_path) == 0:
            self.source_path = base_path

    def find_git_repos(self):
        repos = []
        paths = []
        for root, dirs, files in os.walk(self.source_path):
            for dir in dirs:
                try:
                    repo = Repo(os.path.join(root, dir))
                    if repo:
                        print(f"Found git repo at {os.path.join(root, dir)}")
                        paths.append(os.path.join(root, dir).replace("\.git", ""))
                        repos.append(repo)
                        dirs.remove(dir)  # don't look inside this directory any further
                except InvalidGitRepositoryError:
                    pass  # Not a git repository
        return repos, paths


    def parent_path_contains_substring(self, repo_path, directory):
        for key in repo_path.keys():
            if key in directory:
                return True
        return False


    def path_is_parant(self, path, directory):
        if path in directory:
            return True
        return False


    def find_files(self):
        directories = {}
        for ext in extensions:
            for file_path in glob.glob(f"{self.source_path}/**/*.{ext}", recursive=True):
                directory = os.path.dirname(file_path)  # get the parent directory
                if directory not in directories:
                    if not self.parent_path_contains_substring(directories, directory):
                        directories[directory] = [ext]
                else:
                    if self.parent_path_contains_substring(directories, directory):
                        for each in directories[directory]:
                            if each is not ext:
                                directories[directory].append(ext)
        return directories

    def find_all_files(self):
        all_file_path = []
        for ext in extensions:
            for file_path in glob.glob(f"{self.source_path}/**/*.{ext}", recursive=True):
                all_file_path.append(file_path)
        return all_file_path

    def merge_repo_to_type(self, repo_path, directories):
        new_dict = {}
        for path in repo_path:
            for directory, file_types in directories.items():
                if path not in new_dict and self.path_is_parant(path, directory):
                    new_dict[path] = set()
                    new_dict[path].update(file_types)
                if path in new_dict and self.path_is_parant(path, directory):
                    new_dict[path].update(file_types)
        return new_dict

if __name__ == "__main__":
    source_path = "C:\\Users\\victo\\workspaces\\OpenAI"
    git_util = GitRepoUtils(source_path=source_path)

    # Find all Git repositories within the provided repo_path and its subfolders
    repositories, repo_path = git_util.find_git_repos()

    directories = git_util.find_files()

    print(directories)

    new_dict = git_util.merge_repo_to_type(repo_path, directories)

    for directory, file_types in new_dict.items():
        print(f"Directory: {directory}, File Types: {file_types}")

    all_file_path = git_util.find_all_files()
    for file in all_file_path:
        print(file)