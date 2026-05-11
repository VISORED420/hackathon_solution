import math
import re

# Define the SKILL_ALIASES mapping
SKILL_ALIASES = {
    "python": "python", "pyhton": "python", "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp", "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask", "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker", "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd", "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# Define the resume dataset
resumes = [
    {"name": "Arjun Sharma",    "raw_skills": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"name": "Priya Nair",      "raw_skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"name": "Rahul Gupta",     "raw_skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"name": "Sneha Patel",     "raw_skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"name": "Vikram Singh",    "raw_skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"name": "Ananya Krishnan", "raw_skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"name": "Karan Mehta",     "raw_skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"name": "Deepika Rao",     "raw_skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"name": "Aditya Kumar",    "raw_skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"name": "Meera Iyer",      "raw_skills": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

# Define the job description dataset
job_descriptions = [
    {
        "company": "Kakao", "role": "ML Engineer",
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Visualization"],
        "preferred_skills": ["NLP", "BERT", "Feature Engineering", "Statistics"]
    },
    {
        "company": "Naver", "role": "Backend Engineer",
        "required_skills": ["Java", "Spring Boot", "MySQL", "PostgreSQL", "Microservices", "Docker", "Kubernetes"],
        "preferred_skills": ["REST API", "CI/CD", "Redis"]
    },
    {
        "company": "Line", "role": "Frontend Engineer",
        "required_skills": ["JavaScript", "React", "Vue", "TypeScript", "REST API", "HTML/CSS"],
        "preferred_skills": ["Node.js", "GraphQL", "Redux", "Jest", "AWS"]
    },
]

# Function to normalize skills
def normalize_skills(raw_skills):
    # Split skills on commas, convert to lowercase, and apply alias mapping
    skills = [SKILL_ALIASES.get(skill.lower(), skill) for skill in re.split(r',\s*', raw_skills)]
    # Discard unknown tokens
    skills = [skill for skill in skills if skill in SKILL_ALIASES.values()]
    return skills

# Function to deduplicate skills
def deduplicate_skills(skills):
    # Remove duplicates while preserving order
    unique_skills = []
    for skill in skills:
        if skill not in unique_skills:
            unique_skills.append(skill)
    return unique_skills

# Function to construct TF-IDF vectors
def compute_tfidf(resumes):
    # Compute TF-IDF vectors for each resume
    tfidf_vectors = []
    for resume in resumes:
        # Compute term frequency (TF)
        tf = {skill: 1 / len(resume) for skill in resume}
        # Compute inverse document frequency (IDF)
        idf = {skill: math.log(10 / sum(1 for r in resumes if skill in r)) for skill in set(skill for r in resumes for skill in r)}
        # Compute TF-IDF
        tf_idf = {skill: tf.get(skill, 0) * idf.get(skill, 0) for skill in set(skill for r in resumes for skill in r)}
        tfidf_vectors.append(tf_idf)
    return tfidf_vectors

# Function to build binary vectors for job descriptions
def build_jd_vectors(job_descriptions):
    # Build binary vectors for each job description
    jd_vectors = []
    for jd in job_descriptions:
        # Extract required and preferred skills
        required_skills = [SKILL_ALIASES.get(skill.lower(), skill) for skill in jd['required_skills']]
        preferred_skills = [SKILL_ALIASES.get(skill.lower(), skill) for skill in jd['preferred_skills']]
        # Build binary vector
        vector = {skill: 1 if skill in required_skills or skill in preferred_skills else 0 for skill in set(skill for jd in job_descriptions for skill in [SKILL_ALIASES.get(s.lower(), s) for s in jd['required_skills'] + jd['preferred_skills']])}
        jd_vectors.append(vector)
    return jd_vectors

# Function to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    # Compute dot product
    dot_product = sum(vector1.get(skill, 0) * vector2.get(skill, 0) for skill in set(vector1) | set(vector2))
    # Compute magnitudes
    magnitude1 = math.sqrt(sum(vector1.get(skill, 0) ** 2 for skill in vector1))
    magnitude2 = math.sqrt(sum(vector2.get(skill, 0) ** 2 for skill in vector2))
    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 != 0 else 0
    return similarity

# Main function
def main():
    # Normalize and deduplicate skills in each resume
    normalized_resumes = [deduplicate_skills(normalize_skills(r['raw_skills'])) for r in resumes]
    # Compute TF-IDF vectors for resumes
    tfidf_vectors = compute_tfidf(normalized_resumes)
    # Build binary vectors for job descriptions
    jd_vectors = build_jd_vectors(job_descriptions)
    # Calculate cosine similarity between resumes and job descriptions
    similarities = [[cosine_similarity(tfidf, jd) for jd in jd_vectors] for tfidf in tfidf_vectors]
    # Rank top 3 candidates for each job description
    rankings = []
    for i, jd in enumerate(job_descriptions):
        # Get top 3 candidates with highest similarity
        top_candidates = sorted(enumerate(similarities), key=lambda x: x[1][i], reverse=True)[:3]
        rankings.append([(resumes[candidate[0]]['name'], round(candidate[1][i], 2)) for candidate in top_candidates])
    return rankings

# Run the main function
rankings = main()
for i, ranking in enumerate(rankings):
    print(f"JD-{i+1} — {job_descriptions[i]['company']} ({job_descriptions[i]['role']})")
    print(', '.join(f"{name}({score})" for name, score in ranking))