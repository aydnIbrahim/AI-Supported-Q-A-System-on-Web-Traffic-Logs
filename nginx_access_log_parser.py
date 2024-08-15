"""
nginx_access_log_parser.py

This module provides functionality to parse Nginx access log files and
convert them into a pandas DataFrame. The log format is based on the
common log format used by Nginx.
"""

from datetime import datetime
import re
import pandas as pd


class NginxAccessLogParser:
    """
    A parser for Nginx access logs.

    Attributes:
        log_file_path (str): The file path of the Nginx access log to be parsed.
    """

    def __init__(self, log_file_path: str):
        """
        Initializes the NginxAccessLogParser with the given log file path.

        Args:
            log_file_path (str): The path to the Nginx access log file.
        """
        self.log_file_path = log_file_path
        self.log_pattern = re.compile(
            r'(?P<ip_address>\S+) - - \[(?P<timestamp>[^]]+)] '
            r'"(?P<method>\S+) (?P<url>\S+) (?P<http_protocol>[^"]+)" '
            r'(?P<status>\d{3}) (?P<size>\d+|-) "(?P<referer>[^"]*)" '
            r'"(?P<user_agent>[^"]*)"'
        )

    def parse_log(self) -> pd.DataFrame:
        """
        Parses the Nginx access log file and converts it into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed log entries.
        """
        log_entries = []

        with open(self.log_file_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                match = self.log_pattern.match(line)
                if match:
                    entry = match.groupdict()
                    entry['timestamp'] = datetime.strptime(
                        entry['timestamp'], '%d/%b/%Y:%H:%M:%S %z'
                    )
                    log_entries.append(entry)

        return pd.DataFrame(log_entries)
