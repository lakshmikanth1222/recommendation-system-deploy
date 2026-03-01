# app.py

import json
import re
import os
import traceback
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Ensure uploads directory exists
# Use Vercel's allowed temporary directory for file uploads
UPLOAD_DIR = '/tmp'

# --- SKILLS DICTIONARY FOR RESUME ANALYSIS ---
# A comprehensive list of skills to look for in a resume.
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'html/css', 'react', 'node.js', 'angular', 'vue.js',
    'data analysis', 'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
    'statistical modeling', 'data mining', 'data visualization', 'tableau', 'power bi', 'pandas', 'numpy', 'scipy',
    'matplotlib', 'seaborn', 'keras', 'tensorflow', 'pytorch', 'scikit-learn', 'git', 'docker', 'kubernetes',
    'cloud computing', 'aws', 'azure', 'gcp', 'google cloud platform', 'devops', 'ci/cd', 'jenkins', 'ansible',
    'terraform', 'linux', 'shell scripting', 'bash', 'network security', 'cybersecurity', 'penetration testing',
    'digital forensics', 'agile methodologies', 'scrum', 'project management', 'product management', 'jira',
    'communication', 'teamwork', 'problem solving', 'critical thinking', 'leadership', 'time management',
    'attention to detail', 'creativity', 'emotional intelligence', 'public speaking', 'negotiation',
    'user research', 'ui/ux design', 'figma', 'adobe xd', 'sketch', 'prototyping', 'wireframing', 'user flows',
    'graphic design', 'adobe creative suite', 'photoshop', 'illustrator', 'indesign', 'autocad', 'solidworks',
    'content creation', 'copywriting', 'seo', 'sem', 'social media marketing', 'email marketing', 'google analytics',
    'financial modeling', 'excel', 'vba', 'quantitative analysis', 'risk management', 'investment banking',
    'wealth management', 'accounting', 'auditing', 'supply chain', 'operations management', 'logistics',
    'business development', 'sales', 'crm', 'salesforce', 'customer relationship management', 'hr', 'human resources',
    'talent acquisition', 'recruiting', 'swift', 'kotlin', 'android studio', 'xcode', 'mobile development',
    'restful apis', 'apis', 'graphql', 'microservices', 'data structures', 'algorithms', 'testing', 'qa',
    'quality assurance', 'selenium', 'jest', 'mocha', 'chai', 'circuit design', 'vhdl', 'verilog',
    'medical terminology', 'biology', 'chemistry', 'pharmacology', 'lab techniques', 'healthcare systems (ehr/emr)',
    'typing speed', 'bioinformatics', 'genomics'
]

# Load internship data from the JSON file
try:
    with open('internships.json', 'r', encoding='utf-8') as f:
        internships = json.load(f)
except FileNotFoundError:
    internships = []
    print("FATAL ERROR: internships.json not found. The application cannot run without data.")
except json.JSONDecodeError:
    internships = []
    print("FATAL ERROR: internships.json is not a valid JSON file. Please check for syntax errors.")


# --- AI Model Pre-processing (only if internships were loaded) ---
if internships:
    corpus = []
    for internship in internships:
        text = (f"{internship['title']}. {internship['description']}. "
                f"Required skills are {' '.join(internship['required_skills'])}. "
                f"Sector: {internship['sector']}, Field: {internship['field']}, Branch: {internship.get('branch', '')}.")
        corpus.append(text)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
else:
    tfidf_matrix = None # Ensure this variable exists even if data loading fails

# --- API Endpoints ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve any file from project root (so requests to /index.html, /internships.json, etc. succeed)
@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = os.path.dirname(__file__)
    file_path = os.path.join(root_dir, filename)
    if os.path.isfile(file_path):
        return send_from_directory(root_dir, filename)
    # If file not found, return default 404
    return "Not Found", 404

@app.route('/internships.json')
def serve_internships():
    return send_from_directory('.', 'internships.json')

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    app.logger.info('Received /analyze-resume request')
    try:
        if 'resume' not in request.files:
            app.logger.warning('No resume file in request')
            return jsonify({'error': 'No resume file provided'}), 400
        file = request.files['resume']
        if file.filename == '':
            app.logger.warning('Empty filename received')
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file for debugging
        saved_path = os.path.join(UPLOAD_DIR, file.filename)
        file.stream.seek(0)
        with open(saved_path, 'wb') as f:
            f.write(file.read())
        app.logger.info(f'Saved uploaded resume to {saved_path}')

        # Re-open saved file with PyMuPDF
        pdf_document = fitz.open(saved_path)
        resume_text = "".join(page.get_text() for page in pdf_document)
        pdf_document.close()

        resume_text_lower = resume_text.lower()
        found_skills = set()

        for skill in SKILLS_DB:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_lower):
                # Standardize capitalization for display
                clean_skill = skill.title().replace('Ui/Ux', 'UI/UX').replace('Html/Css', 'HTML/CSS')
                found_skills.add(clean_skill)

        app.logger.info(f'Extracted skills: {found_skills}')
        return jsonify({'skills': sorted(list(found_skills))})

    except Exception as e:
        app.logger.error('Error processing resume: %s', e)
        tb = traceback.format_exc()
        app.logger.error(tb)
        # In debug mode, return traceback to the client to help debugging
        if app.debug:
            return jsonify({'error': 'Failed to process resume. See server logs.', 'traceback': tb}), 500
        return jsonify({'error': 'Failed to process resume. Please ensure it is a valid PDF.'}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    if not internships or tfidf_matrix is None:
        return jsonify({'error': 'Server is not ready. No internship data loaded.'}), 500

    try:
        user_data = request.json
        recommendations = generate_recommendations(user_data)
        return jsonify(recommendations)
    
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return jsonify({'error': str(e)}), 500

def generate_recommendations(user_data):
    education = user_data.get('education', '')
    field = user_data.get('field', '')
    branch = user_data.get('branch', '')
    skills = user_data.get('skills', [])
    sector = user_data.get('sector', '')
    state = user_data.get('state', '')

    user_profile_text = (f"A candidate with skills in {' '.join(skills)}, from the {field} field and {branch} branch, "
                         f"interested in the {sector} sector.")
    
    user_tfidf_vector = tfidf_vectorizer.transform([user_profile_text])
    cosine_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix).flatten()
    
    scored_internships = []
    user_skills_set = set(skill.lower() for skill in skills)

    for i, internship in enumerate(internships):
        content_score = cosine_similarities[i] * 50
        required_skills_set = set(skill.lower() for skill in internship['required_skills'])
        matched_skills = list(user_skills_set.intersection(required_skills_set))
        missing_skills = list(required_skills_set.difference(user_skills_set))
        
        skill_match_percentage = len(matched_skills) / len(required_skills_set) if required_skills_set else 1.0
        skill_score = skill_match_percentage * 35
        
        filter_score = 0
        if education and internship.get('education') in education: filter_score += 5
        if not state or (internship.get('location') in [state, 'Remote']): filter_score += 5
        if branch and internship.get('branch') == branch: filter_score += 5
        
        opportunity_bonus = 5 if 1 <= len(missing_skills) <= 2 and skill_match_percentage >= 0.6 else 0
        
        total_score = content_score + skill_score + filter_score + opportunity_bonus
        
        # Lowered threshold to ensure more matches appear
        if total_score > 30:
            matched_skills_display = [s.replace('_', ' ').title().replace('Ui/Ux', 'UI/UX') for s in matched_skills]
            missing_skills_display = [s.replace('_', ' ').title().replace('Ui/Ux', 'UI/UX') for s in missing_skills]
            
            scored_internships.append({
                **internship,
                'match': min(round(total_score), 99),
                'explainability': {
                    'matched_skills': matched_skills_display,
                    'missing_skills': missing_skills_display,
                    'is_opportunity': opportunity_bonus > 0
                }
            })
    
    scored_internships.sort(key=lambda x: x['match'], reverse=True)
    return scored_internships[:9]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

