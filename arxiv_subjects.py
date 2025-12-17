"""
ArXiv Computer Science 主题配置
包含所有 CS 主题及其对应的 URL 代码
"""

# ArXiv CS 主题映射表
# 格式: "完整主题名称": "URL代码"
ARXIV_CS_SUBJECTS = {
    "Artificial Intelligence": "cs.AI",
    "Computation and Language": "cs.CL",
    "Computational Complexity": "cs.CC",
    "Computational Engineering, Finance, and Science": "cs.CE",
    "Computational Geometry": "cs.CG",
    "Computer Science and Game Theory": "cs.GT",
    "Computer Vision and Pattern Recognition": "cs.CV",
    "Computers and Society": "cs.CY",
    "Cryptography and Security": "cs.CR",
    "Data Structures and Algorithms": "cs.DS",
    "Databases": "cs.DB",
    "Digital Libraries": "cs.DL",
    "Discrete Mathematics": "cs.DM",
    "Distributed, Parallel, and Cluster Computing": "cs.DC",
    "Emerging Technologies": "cs.ET",
    "Formal Languages and Automata Theory": "cs.FL",
    "General Literature": "cs.GL",
    "Graphics": "cs.GR",
    "Hardware Architecture": "cs.AR",
    "Human-Computer Interaction": "cs.HC",
    "Information Retrieval": "cs.IR",
    "Information Theory": "cs.IT",
    "Logic in Computer Science": "cs.LO",
    "Machine Learning": "cs.LG",
    "Mathematical Software": "cs.MS",
    "Multiagent Systems": "cs.MA",
    "Multimedia": "cs.MM",
    "Networking and Internet Architecture": "cs.NI",
    "Neural and Evolutionary Computing": "cs.NE",
    "Numerical Analysis": "cs.NA",
    "Operating Systems": "cs.OS",
    "Other Computer Science": "cs.OH",
    "Performance": "cs.PF",
    "Programming Languages": "cs.PL",
    "Robotics": "cs.RO",
    "Social and Information Networks": "cs.SI",
    "Software Engineering": "cs.SE",
    "Sound": "cs.SD",
    "Symbolic Computation": "cs.SC",
    "Systems and Control": "cs.SY",
}

# 反向映射：代码到主题名称
ARXIV_CODE_TO_SUBJECT = {v: k for k, v in ARXIV_CS_SUBJECTS.items()}


def get_subject_code(subject_name: str) -> str:
    """
    获取主题对应的 ArXiv 代码

    Args:
        subject_name: 主题全称

    Returns:
        ArXiv 代码（如 "cs.AI"），如果未找到则返回 None
    """
    return ARXIV_CS_SUBJECTS.get(subject_name)


def get_subject_name(code: str) -> str:
    """
    获取 ArXiv 代码对应的主题名称

    Args:
        code: ArXiv 代码（如 "cs.AI"）

    Returns:
        主题全称，如果未找到则返回 None
    """
    return ARXIV_CODE_TO_SUBJECT.get(code)


def search_subjects(query: str, limit: int = 10) -> list:
    """
    模糊搜索主题

    Args:
        query: 搜索关键词
        limit: 返回结果数量限制

    Returns:
        匹配的主题列表，每项包含 name 和 code
    """
    query_lower = query.lower()
    results = []

    for name, code in ARXIV_CS_SUBJECTS.items():
        if query_lower in name.lower():
            results.append({"name": name, "code": code})
            if len(results) >= limit:
                break

    return results


def get_all_subjects() -> list:
    """
    获取所有主题列表

    Returns:
        所有主题的列表，每项包含 name 和 code
    """
    return [{"name": name, "code": code} for name, code in ARXIV_CS_SUBJECTS.items()]
