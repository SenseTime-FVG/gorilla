import datetime
import subprocess
from copy import deepcopy
from typing import Dict, List, Optional, Union
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context import (
    FILE_CONTENT_EXTENSION,
    FILES_TAIL_USED,
    POPULATE_FILE_EXTENSION,
)


class File:

    def __init__(self, name: str, content: str = "") -> None:
        """
        Initialize a file with a name and optional content.

        Args:
            name (str): The name of the file.
            content (str, optional): The initial content of the file. Defaults to an empty string.
        """
        self.name: str = name
        self.content: str = content
        self._last_modified: datetime.datetime = datetime.datetime.now()

    def _write(self, new_content: str) -> None:
        """
        Write new content to the file and update the last modified time.

        Args:
            new_content (str): The new content to write to the file.
        """
        self.content = new_content
        self._last_modified = datetime.datetime.now()

    def _read(self) -> str:
        """
        Read the content of the file.

        Returns:
            content (str): The current content of the file.
        """
        return self.content

    def _append(self, additional_content: str) -> None:
        """
        Append content to the existing file content.

        Args:
            additional_content (str): The content to append to the file.
        """
        self.content += additional_content
        self._last_modified = datetime.datetime.now()

    def __repr__(self):
        return f"<<File: {self.name}, Content: {self.content}>>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, File):
            return False
        return self.name == other.name and self.content == other.content


class Directory:

    def __init__(self, name: str, parent: Optional["Directory"] = None) -> None:
        """
        Initialize a directory with a name.

        Args:
            name (str): The name of the directory.
        """
        self.name: str = name
        self.parent: Optional["Directory"] = parent
        self.contents: Dict[str, Union["File", "Directory"]] = {}

    def _add_file(self, file_name: str, content: str = "") -> None:
        """
        Add a new file to the directory.

        Args:
            file_name (str): The name of the file.
            content (str, optional): The content of the new file. Defaults to an empty string.
        """
        if file_name in self.contents:
            raise ValueError(
                f"File '{file_name}' already exists in directory '{self.name}'."
            )
        new_file = File(file_name, content)
        self.contents[file_name] = new_file

    def _add_directory(self, dir_name: str) -> None:
        """
        Add a new subdirectory to the directory.

        Args:
            dir_name (str): The name of the subdirectory.
        """
        if dir_name in self.contents:
            raise ValueError(
                f"Directory '{dir_name}' already exists in directory '{self.name}'."
            )
        new_dir = Directory(dir_name, self)
        self.contents[dir_name] = new_dir

    def _get_item(self, item_name: str) -> Union["File", "Directory", None]:
        """
        Get an item (file or subdirectory) from the directory.

        Args:
            item_name (str): The name of the item to retrieve.

        Returns:
            item (any): The retrieved item or None if it does not exist.
        """
        if item_name == ".":
            return self
        return self.contents.get(item_name)

    def _list_contents(self) -> List[str]:
        """
        List the names of all contents in the directory.

        Returns:
            contents (List[str]): A list of names of the files and subdirectories in the directory.
        """
        return list(self.contents.keys())

    def __repr__(self):
        return f"<Directory: {self.name}, Parent: {self.parent.name if self.parent else None}, Contents: {self.contents}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


DEFAULT_STATE = {"root": Directory("/", None)}


class GorillaFileSystem:

    def __init__(self) -> None:
        """
        Initialize the Gorilla file system with a root directory
        """
        self.root: Directory
        self._current_dir: Directory
        self._api_description = "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GorillaFileSystem):
            return False
        return self.root == other.root

    def _load_scenario(self, scenario: dict, long_context: bool = False) -> None:
        """
        Load a scenario into the file system.

        Args:
            scenario (dict): The scenario to load.

        The scenario always starts with a root directory. Each directory can contain files or subdirectories.
        The key is the name of the file or directory, and the value is a dictionary with the following keys
        An example scenario:
        Here John is the root directory and it contains a home directory with a user directory inside it.
        The user directory contains a file named file1.txt and a directory named directory1.
        Root is not a part of the scenario and it's just easy for parsing. During generation, you should have at most 2 layers.
        {
            "root": {
                "john": {
                    "type": "directory",
                    "contents": {
                        "home": {
                            "type": "directory",
                            "contents": {
                                "user": {
                                    "type": "directory",
                                    "contents": {
                                        "file1.txt": {
                                            "type": "file",
                                            "content": "Hello, world!"
                                        },
                                        "directory1": {
                                            "type": "directory",
                                            "contents": {}
                                        }
                                    }
                                }
                            }
                        }
                }
            }
        }
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.long_context = long_context
        self.root = DEFAULT_STATE_COPY["root"]
        if "root" in scenario:
            root_dir = Directory(list(scenario["root"].keys())[0], None)
            self.root = self._load_directory(
                scenario["root"][list(scenario["root"].keys())[0]]["contents"], root_dir
            )
        self._current_dir = self.root

    def _load_directory(
        self, current: dict, parent: Optional[Directory] = None
    ) -> Directory:
        """
        Load a directory and its contents from a dictionary.

        Args:
            data (dict): The dictionary representing the directory.
            parent (Directory, optional): The parent directory. Defaults to None.

        Returns:
            Directory: The loaded directory.
        """
        is_bottommost = True
        for dir_name, dir_data in current.items():

            if dir_data["type"] == "directory":
                is_bottommost = False
                new_dir = Directory(dir_name, parent)
                new_dir = self._load_directory(dir_data["contents"], new_dir)
                parent.contents[dir_name] = new_dir

            elif dir_data["type"] == "file":
                content = dir_data["content"]
                if self.long_context and dir_name not in FILES_TAIL_USED:
                    content += FILE_CONTENT_EXTENSION
                new_file = File(dir_name, content)
                parent.contents[dir_name] = new_file

        if is_bottommost and self.long_context:
            self._populate_directory(parent)

        return parent

    def _populate_directory(
        self, directory: Directory
    ) -> None:  # Used only for long context
        """
        Populate an innermost directory with multiple empty files.

        Args:
            directory (Directory): The innermost directory to populate.
        """
        for i in range(len(POPULATE_FILE_EXTENSION)):
            name = POPULATE_FILE_EXTENSION[i]
            file_name = f"{name}"
            directory._add_file(file_name)

    def _update_parent_references(self, directory: Directory, new_parent: Directory) -> None:
        """
        Recursively update parent references for a directory and all its subdirectories.

        Args:
            directory (Directory): The directory whose parent references need updating.
            new_parent (Directory): The new parent directory.
        """
        directory.parent = new_parent
        for item in directory.contents.values():
            if isinstance(item, Directory):
                self._update_parent_references(item, directory)

    def pwd(self):
        """
        Return the current working directory path.
        Args:
            None
        Returns:
            current_working_directory (str): The current working directory path.

        """
        path = []
        dir = self._current_dir
        while dir.parent is not None:
            path.append(dir.name)
            dir = dir.parent
        return {"current_working_directory": "/" + "/".join(reversed(path))}

    def ls(self, a: bool = False) -> Dict[str, List[str]]:
        """
        List the contents of the current directory.

        Args:
            a (bool): [Optional] Show hidden files and directories. Defaults to False.

        Returns:
            current_directory_content (List[str]): A list of the contents of the specified directory.
        """
        contents = self._current_dir._list_contents()
        if not a:
            contents = [item for item in contents if not item.startswith(".")]
        return {"current_directory_content": contents}

    def cd(self, folder: str) -> Union[None, Dict[str, str]]:
        """
        Change the current working directory to the specified folder.

        Args:
            folder (str): The folder of the directory to change to. You can only change one folder level at a time.

        Returns:
            current_working_directory (str): The new current working directory path.
        """
        # turn "../" → "..", "/"  → ""
        folder = folder.rstrip("/")
        if folder == "":
            folder = "/"

        # Check if path is nested
        if folder not in {".", "..", "/"} and "/" in folder:
            return {
                "error": f"cd: {folder}: Unsupported path. Only one folder level at a time is supported."
            }

        # Handle navigating to the parent directory with "cd .."
        if folder == "..":
            if self._current_dir.parent:
                self._current_dir = self._current_dir.parent
            elif self.root == self._current_dir:
                return {"error": "Current directory is already the root. Cannot go back."}
            else:
                return {"error": "cd: ..: No such directory"}
            return self.pwd()

        # Handle absolute or relative paths
        target_dir = self._navigate_to_directory(folder)
        if isinstance(target_dir, dict):  # This means there was an error from _navigate_to_directory
            return target_dir
        self._current_dir = target_dir
        return self.pwd()

    def _validate_file_or_directory_name(self, dir_name: str) -> bool:
        if any(c in dir_name for c in '|/\\?%*:"><'):
            return False
        return True

    def mkdir(self, dir_name: str) -> Union[None, Dict[str, str]]:
        """
        Create a new directory in the current directory.

        Args:
            dir_name (str): The name of the new directory at current directory. You can only create directory at current directory.
        """
        if not self._validate_file_or_directory_name(dir_name):
            return {
                "error": f"mkdir: cannot create directory '{dir_name}': Invalid character"
            }
        if dir_name in self._current_dir.contents:
            return {"error": f"mkdir: cannot create directory '{dir_name}': File exists"}

        self._current_dir._add_directory(dir_name)
        return None

    def touch(self, file_name: str) -> Union[None, Dict[str, str]]:
        """
        Create a new file of any extension in the current directory.

        Args:
            file_name (str): The name of the new file in the current directory. file_name is local to the current directory and does not allow path.
        """
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"touch: cannot touch '{file_name}': Invalid character"}

        if file_name in self._current_dir.contents:
            return {"error": f"touch: cannot touch '{file_name}': File exists"}

        self._current_dir._add_file(file_name)
        return None

    def echo(
        self, content: str, file_name: Optional[str] = None
    ) -> Union[Dict[str, str], None]:
        """
        Write content to a file at current directory or display it in the terminal.

        Args:
            content (str): The content to write or display.
            file_name (str): [Optional] The name of the file at current directory to write the content to. Defaults to None.

        Returns:
            terminal_output (str): The content if no file name is provided, or None if written to file.
        """
        if file_name is None:
            return {"terminal_output": content}
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"echo: cannot write to '{file_name}': Invalid character"}

        if file_name:
            if file_name in self._current_dir.contents:
                item = self._current_dir._get_item(file_name)
                if isinstance(item, File):
                    item._write(content)
                    return {"result": f"Content written to '{file_name}'"}
                else:
                    return {"error": f"echo: cannot write to '{file_name}': Is a directory"}
            else:
                self._current_dir._add_file(file_name, content)
                return {"result": f"Content written to '{file_name}'"}
        else:
            return {"terminal_output": content}

    def cat(self, file_name: str) -> Dict[str, str]:
        """
        Display the contents of a file of any extension from currrent directory.

        Args:
            file_name (str): The name of the file from current directory to display. No path is allowed.

        Returns:
            file_content (str): The content of the file.
        """
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"cat: '{file_name}': Invalid character"}

        if file_name in self._current_dir.contents:
            item = self._current_dir._get_item(file_name)
            if isinstance(item, File):
                return {"file_content": item._read()}
            else:
                return {"error": f"cat: '{file_name}': Is a directory"}
        else:
            return {"error": f"cat: '{file_name}': No such file or directory"}

    def find(self, path: str = ".", name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Find any file or directories under specific path that contain name in its file name.

        This method searches for files of any extension and directories within a specified path that match
        the given name. If no name is provided, it returns all files and directories
        in the specified path and its subdirectories.
        Note: This method performs a recursive search through all subdirectories of the given path.

        Args:
            path (str): The directory path to start the search. Defaults to the current directory (".").
            name (str): [Optional] The name of the file or directory to search for. If None, all items are returned.

        Returns:
            matches (List[str]): A list of matching file and directory paths relative to the given path.

        """
        matches = []
        # Navigate to the requested path first
        target_dir = self._navigate_to_directory(path)
        if isinstance(target_dir, dict):  # invalid path
            # Replace the tool name in the error message for clarity
            original_msg = target_dir.get("error", "")
            # e.g. "cd: '/foo': No such file or directory" -> "find: '/foo': No such file or directory"
            if original_msg.startswith("cd:"):
                return {"error": original_msg.replace("cd:", "find:", 1)}
            return target_dir

        def recursive_search(directory: Directory, base_path: str) -> None:
            for item_name, item in directory.contents.items():
                item_path = f"{base_path}/{item_name}"
                if name is None or name in item_name:
                    matches.append(item_path)
                if isinstance(item, Directory):
                    recursive_search(item, item_path)

        recursive_search(target_dir, path.rstrip("/"))
        return {"matches": matches}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Union[int, str]]:
        """
        Count the number of lines, words, and characters in a file of any extension from current directory.

        Args:
            file_name (str): Name of the file of current directory to perform wc operation on.
            mode (str): Mode of operation ('l' for lines, 'w' for words, 'c' for characters).

        Returns:
            count (int): The count of the number of lines, words, or characters in the file.
            type (str): The type of unit we are counting. [Enum]: ["lines", "words", "characters"]
        """
        if mode not in ["l", "w", "c"]:
            return {"error": f"wc: invalid mode '{mode}'"}

        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                if mode == "l":
                    line_count = len(content.splitlines())
                    return {"count": line_count, "type": "lines"}

                elif mode == "w":
                    word_count = len(content.split())
                    return {"count": word_count, "type": "words"}

                elif mode == "c":
                    char_count = len(content)
                    return {"count": char_count, "type": "characters"}

        return {"error": f"wc: {file_name}: No such file or directory"}

    def sort(self, file_name: str) -> Dict[str, str]:
        """
        Sort the contents of a file line by line.

        Args:
            file_name (str): The name of the file appeared at current directory to sort.

        Returns:
            sorted_content (str): The sorted content of the file.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                sorted_content = "\n".join(sorted(content.splitlines()))

                return {"sorted_content": sorted_content}

        return {"error": f"sort: {file_name}: No such file or directory"}

    def grep(self, file_name: str, pattern: str) -> Dict[str, List[str]]:
        """
        Search for lines in a file of any extension at current directory that contain the specified pattern.

        Args:
            file_name (str): The name of the file to search. No path is allowed and you can only perform on file at local directory.
            pattern (str): The pattern to search for.

        Returns:
            matching_lines (List[str]): Lines that match the pattern.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                matching_lines = [line for line in content.splitlines() if pattern in line]

                return {"matching_lines": matching_lines}

        return {"error": f"grep: {file_name}: No such file or directory"}

    def du(self, human_readable: bool = False) -> Dict[str, str]:
        """
        Estimate the disk usage of a directory and its contents.

        Args:
            human_readable (bool): If True, returns the size in human-readable format (e.g., KB, MB).

        Returns:
            disk_usage (str): The estimated disk usage.
        """

        def get_size(item: Union[File, Directory]) -> int:
            if isinstance(item, File):
                return len(item._read().encode("utf-8"))
            elif isinstance(item, Directory):
                return sum(get_size(child) for child in item.contents.values())
            return 0

        target_dir = self._navigate_to_directory(None)
        if isinstance(target_dir, dict):  # Error condition check
            return target_dir

        total_size = get_size(target_dir)

        if human_readable:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if total_size < 1024:
                    size_str = f"{total_size:.2f} {unit}"
                    break
                total_size /= 1024
            else:
                size_str = f"{total_size:.2f} PB"
        else:
            size_str = f"{total_size} bytes"

        return {"disk_usage": size_str}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, str]:
        """
        Display the last part of a file of any extension.

        Args:
            file_name (str): The name of the file to display. No path is allowed and you can only perform on file at local directory.
            lines (int): The number of lines to display from the end of the file. Defaults to 10.

        Returns:
            last_lines (str): The last part of the file.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read().splitlines()

                if lines > len(content):
                    lines = len(content)

                last_lines = content[-lines:]
                return {"last_lines": "\n".join(last_lines)}

        return {"error": f"tail: {file_name}: No such file or directory"}

    def diff(self, file_name1: str, file_name2: str) -> Dict[str, str]:
        """
        Compare two files of any extension line by line at the current directory.

        Args:
            file_name1 (str): The name of the first file in current directory.
            file_name2 (str): The name of the second file in current directory.

        Returns:
            diff_lines (str): The differences between the two files.
        """
        if (
            file_name1 in self._current_dir.contents
            and file_name2 in self._current_dir.contents
        ):
            file1 = self._current_dir._get_item(file_name1)
            file2 = self._current_dir._get_item(file_name2)

            if isinstance(file1, File) and isinstance(file2, File):
                content1 = file1._read().splitlines()
                content2 = file2._read().splitlines()

                diff_lines = [
                    f"- {line1}\n+ {line2}"
                    for line1, line2 in zip(content1, content2)
                    if line1 != line2
                ]

                return {"diff_lines": "\n".join(diff_lines)}

        return {"error": f"diff: {file_name1} or {file_name2}: No such file or directory"}

    def mv(self, source: str, destination: str) -> Dict[str, str]:
        """
        Move a file or directory from one location to another.

        Args:
            source (str): Source name of the file or directory to move. Source must be local to the current directory.
            destination (str): The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name.

        Returns:
            result (str): The result of the move operation.
        """
        if source not in self._current_dir.contents:
            return {"error": f"mv: cannot move '{source}': No such file or directory"}

        item = self._current_dir._get_item(source)

        if not isinstance(item, (File, Directory)):
            return {"error": f"mv: cannot move '{source}': Not a file or directory"}

        if "/" in destination:
            return {
                "error": "mv: path not allowed in destination. Provide only a file or directory name."
            }

        # Check if the destination is an existing directory
        if destination in self._current_dir.contents:
            dest_item = self._current_dir._get_item(destination)
            if isinstance(dest_item, Directory):
                # Move source into the destination directory
                new_destination = f"{source}"
                if new_destination in dest_item.contents:
                    return {
                        "error": f"mv: cannot move '{source}' to '{destination}/{source}': File exists"
                    }
                else:
                    self._current_dir.contents.pop(source)
                    if isinstance(item, File):
                        dest_item._add_file(source, item.content)
                    else:
                        dest_item._add_directory(source)
                        dest_item.contents[source].contents = deepcopy(item.contents)
                        self._update_parent_references(dest_item.contents[source], dest_item)
                    return {"result": f"'{source}' moved to '{destination}/{source}'"}
            else:
                return {
                    "error": f"mv: cannot move '{source}' to '{destination}': Not a directory"
                }
        else:
            # Destination is not an existing directory, move/rename the item
            self._current_dir.contents.pop(source)
            if isinstance(item, File):
                self._current_dir._add_file(destination, item.content)
            else:
                self._current_dir._add_directory(destination)
                self._current_dir.contents[destination].contents = deepcopy(item.contents)
                self._update_parent_references(self._current_dir.contents[destination], self._current_dir)
            return {"result": f"'{source}' moved to '{destination}'"}

    def rm(self, file_name: str) -> Dict[str, str]:
        """
        Remove a file or directory.

        Args:
            file_name (str): The name of the file or directory to remove.

        Returns:
            result (str): The result of the remove operation.
        """
        if file_name in self._current_dir.contents:
            item = self._current_dir._get_item(file_name)
            if isinstance(item, File) or isinstance(item, Directory):
                self._current_dir.contents.pop(file_name)
                return {"result": f"'{file_name}' removed"}
            else:
                return {
                    "error": f"rm: cannot remove '{file_name}': Not a file or directory"
                }
        else:
            return {"error": f"rm: cannot remove '{file_name}': No such file or directory"}

    def rmdir(self, dir_name: str) -> Dict[str, str]:
        """
        Remove a directory at current directory.

        Args:
            dir_name (str): The name of the directory to remove. Directory must be local to the current directory.

        Returns:
            result (str): The result of the remove operation.
        """
        if dir_name in self._current_dir.contents:
            item = self._current_dir._get_item(dir_name)
            if isinstance(item, Directory):
                if item.contents:  # Check if directory is not empty
                    return {
                        "error": f"rmdir: cannot remove '{dir_name}': Directory not empty"
                    }
                else:
                    self._current_dir.contents.pop(dir_name)
                    return {"result": f"'{dir_name}' removed"}
            else:
                return {"error": f"rmdir: cannot remove '{dir_name}': Not a directory"}
        else:
            return {
                "error": f"rmdir: cannot remove '{dir_name}': No such file or directory"
            }

    def cp(self, source: str, destination: str) -> Dict[str, str]:
        """
        Copy a file or directory from one location to another.

        If the destination is a directory, the source file or directory will be copied
        into the destination directory.

        Both source and destination must be local to the current directory.

        Args:
            source (str): The name of the file or directory to copy.
            destination (str): The destination name to copy the file or directory to.
                            If the destination is a directory, the source will be copied
                            into this directory. No file paths allowed.

        Returns:
            result (str): The result of the copy operation or an error message if the operation fails.
        """
        if source not in self._current_dir.contents:
            return {"error": f"cp: cannot copy '{source}': No such file or directory"}

        item = self._current_dir._get_item(source)

        if not isinstance(item, (File, Directory)):
            return {"error": f"cp: cannot copy '{source}': Not a file or directory"}

        if "/" in destination:
            return {
                "error": "cp: path not allowed in destination. Provide only a file or directory name."
            }
        # Check if the destination is an existing directory
        if destination in self._current_dir.contents:
            dest_item = self._current_dir._get_item(destination)
            if isinstance(dest_item, Directory):
                # Copy source into the destination directory
                # Inside dest_item, the key should be just the source name
                if source in dest_item.contents:
                    return {
                        "error": f"cp: cannot copy '{source}' to '{destination}/{source}': File exists"
                    }
                else:
                    if isinstance(item, File):
                        dest_item._add_file(source, item.content)
                    else:
                        dest_item._add_directory(source)
                        dest_item.contents[source].contents = deepcopy(item.contents)
                        self._update_parent_references(dest_item.contents[source], dest_item)
                    return {"result": f"'{source}' copied to '{destination}/{source}'"}
            else:
                return {
                    "error": f"cp: cannot copy '{source}' to '{destination}': Not a directory"
                }
        else:
            # Destination is not an existing directory, perform the copy
            if isinstance(item, File):
                self._current_dir._add_file(destination, item.content)
            else:
                self._current_dir._add_directory(destination)
                self._current_dir.contents[destination].contents = deepcopy(item.contents)
                self._update_parent_references(self._current_dir.contents[destination], self._current_dir)
            return {"result": f"'{source}' copied to '{destination}'"}

    def _navigate_to_directory(
        self, path: Optional[str]
    ) -> Union[Directory, Dict[str, str]]:
        """
        Navigate to a specified directory path from the current directory.

        Args:
            path (str): [Optional] The path to navigate to. Defaults to None (current directory).

        Returns:
            target_directory (Directory or dict): The target directory object or error message.
        """
        if path is None or path == ".":
            return self._current_dir
        elif path == "/":
            return self.root

        dirs = path.strip("/").split("/")
        temp_dir = self._current_dir if not path.startswith("/") else self.root

        for dir_name in dirs:
            next_dir = temp_dir._get_item(dir_name)
            if isinstance(next_dir, Directory):
                temp_dir = next_dir
            else:
                return {"error": f"cd: '{path}': No such file or directory"}

        return temp_dir

    def _parse_positions(self, positions: str) -> List[int]:
        """
        Helper function to parse position strings, e.g., '1,3,5', '1-5', '-3', or '3-'.

        Args:
            positions (str): The position string to parse.

        Returns:
            list (List[int]): A list of integers representing the positions.
        """
        result = []
        if "," in positions:
            for part in positions.split(","):
                result.extend(self._parse_positions(part))
        elif "-" in positions:
            start, end = positions.split("-")
            start = int(start) if start else 1
            # If end is empty (e.g. "3-"), treat it as a single position "start"
            # instead of trying to create an unbounded range which could lead
            # to extremely large memory usage.
            if end:
                end = int(end)
                result.extend(range(start, end + 1))
            else:
                result.append(start)
        else:
            result.append(int(positions))
        return result


def normal_test():
    """
    普通功能的综合测试：验证文件系统的基础行为是否正常。
    """
    print("=" * 60)
    print("开始执行 normal_test（基础功能测试）")
    print("=" * 60)
    
    # 初始化文件系统
    fs = GorillaFileSystem()
    fs.root = Directory("/", None)
    fs._current_dir = fs.root
    
    # 测试 1: pwd - 查看当前目录
    print("\n[测试 1] pwd - 查看当前目录")
    result = fs.pwd()
    print(f"结果: {result}")
    assert result["current_working_directory"] == "/"
    
    # 测试 2: mkdir - 创建目录
    print("\n[测试 2] mkdir - 创建目录")
    result = fs.mkdir("test_dir")
    print(f"创建 test_dir: {result}")
    assert result is None
    
    result = fs.mkdir("another_dir")
    print(f"创建 another_dir: {result}")
    assert result is None
    
    # 测试错误情况：创建已存在的目录
    result = fs.mkdir("test_dir")
    print(f"尝试创建已存在的目录: {result}")
    assert "error" in result
    
    # 测试 3: ls - 列出目录内容
    print("\n[测试 3] ls - 列出目录内容")
    result = fs.ls()
    print(f"当前目录内容: {result}")
    assert "test_dir" in result["current_directory_content"]
    assert "another_dir" in result["current_directory_content"]
    
    # 测试 4: touch - 创建文件
    print("\n[测试 4] touch - 创建文件")
    result = fs.touch("file1.txt")
    print(f"创建 file1.txt: {result}")
    assert result is None
    
    result = fs.touch("file2.txt")
    print(f"创建 file2.txt: {result}")
    assert result is None
    
    # 测试 5: echo - 写入文件内容
    print("\n[测试 5] echo - 写入文件内容")
    result = fs.echo("Hello, World!", "file1.txt")
    print(f"写入 file1.txt: {result}")
    assert result == {"result": "Content written to 'file1.txt'"}

    result = fs.echo("Line 1\nLine 2\nLine 3", "file2.txt")
    print(f"写入 file2.txt: {result}")
    assert result == {"result": "Content written to 'file2.txt'"}
    
    # 测试 echo 输出到终端
    result = fs.echo("This is terminal output")
    print(f"echo 到终端: {result}")
    assert "terminal_output" in result
    
    # 测试 6: cat - 读取文件内容
    print("\n[测试 6] cat - 读取文件内容")
    result = fs.cat("file1.txt")
    print(f"读取 file1.txt: {result}")
    assert result["file_content"] == "Hello, World!"
    
    result = fs.cat("file2.txt")
    print(f"读取 file2.txt: {result}")
    assert "Line 1" in result["file_content"]
    
    # 测试错误情况：读取不存在的文件
    result = fs.cat("nonexistent.txt")
    print(f"读取不存在的文件: {result}")
    assert "error" in result
    
    # 测试 7: cd - 切换目录并验证返回完整路径
    print("\n[测试 7] cd - 切换目录")
    result = fs.cd("test_dir")
    print(f"切换到 test_dir: {result}")
    assert result == {"current_working_directory": "/test_dir"}

    result = fs.pwd()
    print(f"当前目录: {result}")
    assert result == {"current_working_directory": "/test_dir"}

    # 在子目录中创建文件
    fs.touch("subfile.txt")
    fs.echo("Subdirectory content", "subfile.txt")

    # 测试多层目录结构
    fs.mkdir("level2")
    fs.cd("level2")
    result = fs.pwd()
    print(f"二级目录 pwd: {result}")
    assert result == {"current_working_directory": "/test_dir/level2"}

    fs.mkdir("level3")
    fs.cd("level3")
    result = fs.pwd()
    print(f"三级目录 pwd: {result}")
    assert result == {"current_working_directory": "/test_dir/level2/level3"}

    # 返回父目录
    result = fs.cd("..")
    print(f"返回父目录: {result}")
    assert result == {"current_working_directory": "/test_dir/level2"}

    fs.cd("..")
    fs.cd("..")
    result = fs.pwd()
    print(f"返回根目录: {result}")
    assert result == {"current_working_directory": "/"}
    
    # 测试 8: wc - 统计文件
    print("\n[测试 8] wc - 统计文件")
    result = fs.wc("file2.txt", "l")
    print(f"统计 file2.txt 行数: {result}")
    assert result["count"] == 3
    assert result["type"] == "lines"
    
    result = fs.wc("file2.txt", "w")
    print(f"统计 file2.txt 单词数: {result}")
    assert result["type"] == "words"
    
    result = fs.wc("file2.txt", "c")
    print(f"统计 file2.txt 字符数: {result}")
    assert result["type"] == "characters"
    
    # 测试 9: sort - 排序文件
    print("\n[测试 9] sort - 排序文件")
    fs.echo("zebra\napple\nbanana", "unsorted.txt")
    result = fs.sort("unsorted.txt")
    print(f"排序 unsorted.txt: {result}")
    assert "apple" in result["sorted_content"]
    assert result["sorted_content"].split("\n")[0] == "apple"
    
    # 测试 10: grep - 搜索文件
    print("\n[测试 10] grep - 搜索文件")
    result = fs.grep("file2.txt", "Line")
    print(f"在 file2.txt 中搜索 'Line': {result}")
    assert len(result["matching_lines"]) > 0
    
    # 测试 11: tail - 查看文件尾部
    print("\n[测试 11] tail - 查看文件尾部")
    result = fs.tail("file2.txt", 2)
    print(f"查看 file2.txt 最后 2 行: {result}")
    assert "Line 2" in result["last_lines"] or "Line 3" in result["last_lines"]
    
    # 测试 12: find - 查找文件
    print("\n[测试 12] find - 查找文件")
    result = fs.find(".", "file")
    print(f"查找包含 'file' 的文件: {result}")
    assert len(result["matches"]) > 0
    
    result = fs.find("test_dir")
    print(f"查找 test_dir 下的所有文件: {result}")
    assert "subfile.txt" in str(result["matches"])
    
    # 测试 13: diff - 比较文件
    print("\n[测试 13] diff - 比较文件")
    fs.echo("same content", "file3.txt")
    fs.echo("same content", "file4.txt")
    result = fs.diff("file3.txt", "file4.txt")
    print(f"比较相同文件: {result}")
    
    fs.echo("different", "file5.txt")
    result = fs.diff("file3.txt", "file5.txt")
    print(f"比较不同文件: {result}")
    assert "diff_lines" in result
    
    # 测试 14: mv - 移动/重命名
    print("\n[测试 14] mv - 移动/重命名")
    result = fs.mv("file1.txt", "renamed_file.txt")
    print(f"重命名 file1.txt: {result}")
    assert "result" in result

    # 验证文件已重命名
    result = fs.cat("renamed_file.txt")
    print(f"读取重命名后的文件: {result}")
    assert result["file_content"] == "Hello, World!"

    # 测试移动目录并验证 parent 引用
    print("\n[测试 14.1] mv - 移动目录并验证 parent 引用")
    fs.mkdir("move_test_dir")
    fs.cd("move_test_dir")
    fs.mkdir("inner_dir")
    fs.cd("inner_dir")
    fs.mkdir("deep_dir")
    fs.cd("..")
    fs.cd("..")

    # 移动目录
    result = fs.mv("move_test_dir", "moved_test_dir")
    print(f"移动目录: {result}")
    assert "result" in result

    # 进入移动后的目录，验证 pwd 正确
    fs.cd("moved_test_dir")
    fs.cd("inner_dir")
    result = fs.pwd()
    print(f"移动后目录中的 pwd: {result}")
    assert result == {"current_working_directory": "/moved_test_dir/inner_dir"}

    fs.cd("deep_dir")
    result = fs.pwd()
    print(f"移动后深层目录中的 pwd: {result}")
    assert result == {"current_working_directory": "/moved_test_dir/inner_dir/deep_dir"}

    fs.cd("..")
    fs.cd("..")
    fs.cd("..")

    # 测试移动到目录
    result = fs.mv("file2.txt", "test_dir")
    print(f"移动 file2.txt 到 test_dir: {result}")
    assert "result" in result

    # 测试 15: cp - 复制文件
    print("\n[测试 15] cp - 复制文件")
    result = fs.cp("renamed_file.txt", "copied_file.txt")
    print(f"复制文件: {result}")
    assert "result" in result

    result = fs.cat("copied_file.txt")
    print(f"读取复制的文件: {result}")
    assert result["file_content"] == "Hello, World!"

    # 测试复制目录并验证 parent 引用
    print("\n[测试 15.1] cp - 复制目录并验证 parent 引用")
    fs.mkdir("copy_test_dir")
    fs.cd("copy_test_dir")
    fs.mkdir("inner_copy")
    fs.cd("inner_copy")
    fs.mkdir("deep_copy")
    fs.cd("..")
    fs.cd("..")

    # 复制目录
    result = fs.cp("copy_test_dir", "copied_test_dir")
    print(f"复制目录: {result}")
    assert "result" in result

    # 进入复制后的目录，验证 pwd 正确
    fs.cd("copied_test_dir")
    fs.cd("inner_copy")
    result = fs.pwd()
    print(f"复制后目录中的 pwd: {result}")
    assert result == {"current_working_directory": "/copied_test_dir/inner_copy"}

    fs.cd("deep_copy")
    result = fs.pwd()
    print(f"复制后深层目录中的 pwd: {result}")
    assert result == {"current_working_directory": "/copied_test_dir/inner_copy/deep_copy"}

    fs.cd("..")
    fs.cd("..")
    fs.cd("..")

    # 测试复制到目录
    result = fs.cp("copied_file.txt", "another_dir")
    print(f"复制文件到目录: {result}")
    assert "result" in result
    
    # 测试 16: du - 磁盘使用情况
    print("\n[测试 16] du - 磁盘使用情况")
    result = fs.du()
    print(f"磁盘使用情况（字节）: {result}")
    assert "disk_usage" in result
    
    result = fs.du(human_readable=True)
    print(f"磁盘使用情况（可读格式）: {result}")
    assert "disk_usage" in result
    
    # 测试 17: rm - 删除文件
    print("\n[测试 17] rm - 删除文件")
    result = fs.rm("copied_file.txt")
    print(f"删除文件: {result}")
    assert "result" in result
    
    # 验证文件已删除
    result = fs.cat("copied_file.txt")
    print(f"尝试读取已删除的文件: {result}")
    assert "error" in result
    
    # 测试 18: rmdir - 删除目录
    print("\n[测试 18] rmdir - 删除目录")
    # 先创建一个空目录
    fs.mkdir("empty_dir")
    result = fs.rmdir("empty_dir")
    print(f"删除空目录: {result}")
    assert "result" in result
    
    # 测试删除非空目录（应该失败）
    result = fs.rmdir("test_dir")
    print(f"尝试删除非空目录: {result}")
    assert "error" in result
    
    # 测试 19: 测试错误处理
    print("\n[测试 19] 错误处理测试")
    
    # 测试无效字符
    result = fs.touch("file/name.txt")
    print(f"创建包含无效字符的文件名: {result}")
    assert "error" in result
    
    # 测试不支持的路径
    result = fs.cd("test_dir/subdir")
    print(f"尝试多级路径切换: {result}")
    assert "error" in result
    
    # 测试 ls -a (显示隐藏文件)
    fs.touch(".hidden_file")
    result = fs.ls()
    print(f"ls (不显示隐藏文件): {result}")
    assert ".hidden_file" not in result["current_directory_content"]
    
    result = fs.ls(a=True)
    print(f"ls -a (显示隐藏文件): {result}")
    assert ".hidden_file" in result["current_directory_content"]
    
    print("\n" + "=" * 60)
    print("normal_test：所有基础功能测试完成！✓")
    print("=" * 60)


def regression_test():
    """
    回归测试：专门验证之前存在但已修复的 bug。
    1. cp 复制到目录时，目录中已存在同名文件应报错，而不是静默覆盖。
    2. _parse_positions 处理 '3-' 不应创建无限范围，而是安全返回 [3]。
    """
    print("=" * 60)
    print("开始执行 regression_test（修复点回归测试）")
    print("=" * 60)

    # 回归测试 1：cp 到已有同名文件的目录
    print("\n[回归测试 1] cp 复制到已有同名文件的目录时应报错")
    fs = GorillaFileSystem()
    fs.root = Directory("/", None)
    fs._current_dir = fs.root

    # 在当前目录创建源文件和目标目录
    fs.echo("source content", "src.txt")
    fs.mkdir("dst_dir")

    # 在目标目录中手动创建一个同名文件，模拟“已存在”
    dst_dir = fs._current_dir._get_item("dst_dir")
    assert isinstance(dst_dir, Directory)
    dst_dir._add_file("src.txt", "existing content")

    # 执行 cp，期望返回错误而不是成功
    result = fs.cp("src.txt", "dst_dir")
    print(f"执行 cp('src.txt', 'dst_dir') 的结果: {result}")
    assert "error" in result
    assert "File exists" in result["error"]

    # 回归测试 2：_parse_positions 安全处理 '3-' 这类输入
    print("\n[回归测试 2] _parse_positions 安全处理 '3-'")
    fs2 = GorillaFileSystem()
    # 该方法不依赖文件系统状态，只测试解析逻辑
    positions1 = fs2._parse_positions("1-5")
    print(f"_parse_positions('1-5') = {positions1}")
    assert positions1 == [1, 2, 3, 4, 5]

    positions2 = fs2._parse_positions("3-")
    print(f"_parse_positions('3-') = {positions2}")
    # 修复后的期望行为：返回 [3]，而不是尝试构造无限范围
    assert positions2 == [3]

    positions3 = fs2._parse_positions("1,3-4")
    print(f"_parse_positions('1,3-4') = {positions3}")
    assert positions3 == [1, 3, 4]

    print("\n" + "=" * 60)
    print("regression_test：所有回归测试通过！✓")
    print("=" * 60)


def main():
    """
    入口函数：先跑基础功能测试，再跑回归测试。
    """
    normal_test()
    regression_test()


if __name__ == "__main__":
    main()
