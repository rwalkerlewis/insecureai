"""
Module 8: Hackathon Interactive Chat with LLM-Based Classification
Learn: Interactive Q&A with the model, enhanced with LLM-based prompt screening
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists
import numpy as np
import re
from typing import Dict

# ============================================================================
# LLM-BASED CLASSIFICATION (adapted from Module 09 approach)
# ============================================================================

class LLMClassifier:
    """
    Classify prompts using the LLM directly instead of embeddings
    This is faster and doesn't require Ollama API
    """
    
    def __init__(self, categories: Dict[str, str], llm: Llama):
        """
        Initialize classifier with categories
        
        Args:
            categories: Dict mapping category names to descriptions
            llm: Llama model instance to use for classification
        """
        self.categories = categories
        self.llm = llm
        
        print(f"🔄 Initializing LLM-based Classifier with {len(categories)} categories...")
        print(f"✅ Classifier ready\n")
    
    def classify(self, prompt: str) -> Dict:
        """
        Classify a prompt using LLM to determine which category it belongs to
        
        Args:
            prompt: Input text to classify
            
        Returns:
            Dictionary with classification results including category and confidence
        """
        # Build simplified classification prompt
        category_list = "\n".join([f"{i+1}. {name}" 
                                   for i, name in enumerate(self.categories.keys())])
        
        classification_prompt = f"""Classify this question into ONE category:

{category_list}

Question: {prompt}

Category name:"""
        
        # Get classification from LLM
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise classifier. Respond with only the category name."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=10,  # Reduced from 30
            temperature=0.0,  # Deterministic for speed
            logprobs=True,
            top_logprobs=5
        )
        
        category_response = response['choices'][0]['message']['content'].strip().lower()
        
        # Extract confidence from first token probability
        confidence = 0.5  # default
        logprobs_data = response['choices'][0].get('logprobs', {})
        if logprobs_data and 'content' in logprobs_data:
            first_tokens = logprobs_data['content'][:2]  # First 2 tokens only
            probs = [np.exp(t['logprob']) for t in first_tokens if t and 'logprob' in t]
            if probs:
                confidence = np.mean(probs)
        
        # Match to actual category name
        matched_category = None
        for cat_name in self.categories.keys():
            if cat_name in category_response or category_response in cat_name:
                matched_category = cat_name
                break
        
        # If no match found, use pattern matching on response
        if not matched_category:
            if any(word in category_response for word in ['fact', 'historical', 'definition', 'math']):
                matched_category = 'safe-factual'
            elif any(word in category_response for word in ['how', 'instruction', 'tutorial']):
                matched_category = 'safe-instructional'
            elif any(word in category_response for word in ['current', 'today', 'now', 'latest']):
                matched_category = 'risky-temporal'
            elif any(word in category_response for word in ['future', 'will', 'predict']):
                matched_category = 'risky-future'
            elif any(word in category_response for word in ['personal', 'private', 'password']):
                matched_category = 'risky-personal'
            elif any(word in category_response for word in ['creative', 'story', 'poem']):
                matched_category = 'risky-creative'
            elif any(word in category_response for word in ['opinion', 'subjective', 'prefer']):
                matched_category = 'risky-opinion'
            elif any(word in category_response for word in ['random', 'choice']):
                matched_category = 'risky-random'
            else:
                matched_category = 'safe-factual'  # default fallback
        
        return {
            'category': matched_category,
            'confidence': confidence,
            'raw_response': category_response
        }


# Define screening categories for prompt classification
SCREENING_CATEGORIES = {
    "safe-factual": "factual questions with objective answers that can be verified, historical facts, definitions, scientific facts, mathematical calculations",
    
    "safe-instructional": "how-to questions asking for step-by-step instructions, tutorials, explanations of processes and methods",
    
    "risky-temporal": "questions asking about current events, real-time information, today's weather, current stock prices, latest news that require up-to-date data",
    
    "risky-future": "questions about future events, predictions, what will happen next, forecasting future outcomes",
    
    "risky-personal": "questions asking for personal private information, user's name, address, password, or other confidential data",
    
    "risky-creative": "creative writing requests, storytelling, poetry, jokes, fictional narratives with infinite valid outputs",
    
    "risky-opinion": "subjective questions asking for opinions, preferences, recommendations, value judgments with no objective answer",
    
    "risky-random": "requests for random selections, arbitrary choices, random number generation"
}


# ============================================================================
# UNCERTAINTY ANALYSIS (from Module 08)
# ============================================================================

def categorize_question_uncertainty(question):
    """
    Analyze the question BEFORE running the model to determine if it inherently
    has maximum epistemic or aleatoric uncertainty.
    
    Returns:
        dict with 'should_run' (bool), 'reason' (str), 'epistemic_max' (bool), 
        'aleatoric_max' (bool), 'category' (str)
    """
    
    question_lower = question.lower().strip()
    
    # Check for MAXIMUM EPISTEMIC UNCERTAINTY cases
    # These are questions where NO model can know the answer (unknowable/future/real-time)
    
    # 1. Real-time information requests (current weather, stock prices, news)
    real_time_patterns = [
        r'\b(current|now|today|right now|at the moment)\b.*\b(weather|temperature|forecast)\b',
        r'\b(what is|what\'s)\b.*\b(weather|temperature)\b.*\b(today|now|currently)\b',
        r'\bstock price',
        r'\bcurrent (news|events)',
        r'\blive (score|update)',
        r'\btoday\'s (news|weather|events)',
    ]
    
    for pattern in real_time_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question requires real-time/current information that no static model can know',
                'epistemic_max': True,
                'aleatoric_max': False,
                'category': 'UNKNOWABLE - Real-time Data Required',
                'explanation': 'This question asks for current/live information (weather, stocks, news, etc.). '
                              'Language models are trained on historical data and cannot access real-time information. '
                              'Epistemic uncertainty is 100% - the model fundamentally cannot know this answer.'
            }
    
    # 2. Future predictions/events
    future_patterns = [
        r'\bwill\b.*\b(happen|be|occur)\b',
        r'\bgoing to\b',
        r'\bnext (year|month|week)',
        r'\bfuture\b',
        r'\bpredict\b',
        r'\bin \d+ (years|months|days)',
    ]
    
    for pattern in future_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question asks about future events which are inherently unpredictable',
                'epistemic_max': True,
                'aleatoric_max': False,
                'category': 'UNKNOWABLE - Future Event',
                'explanation': 'This question asks about future events. No model (or human) can know '
                              'future outcomes with certainty. Epistemic uncertainty is 100% - the answer '
                              'does not exist yet and cannot be known.'
            }
    
    # 3. Personal/private information about the user
    personal_patterns = [
        r'\bmy (name|age|address|phone|email|password)\b',
        r'\bwho am i\b',
        r'\bwhere do i live\b',
        r'\bwhat is my\b',
    ]
    
    for pattern in personal_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question asks for personal information about you that the model cannot know',
                'epistemic_max': True,
                'aleatoric_max': False,
                'category': 'UNKNOWABLE - Private Personal Data',
                'explanation': 'This question asks for personal information about you. The model has no '
                              'access to private user data. Epistemic uncertainty is 100% - this information '
                              'is unknowable without external context.'
            }
    
    # Check for MAXIMUM ALEATORIC UNCERTAINTY cases
    # These are questions with infinite valid answers (creative/subjective/opinion)
    
    # 1. Creative writing/generation
    creative_patterns = [
        r'\bwrite (me )?(a|an) (story|poem|song|joke)\b',
        r'\btell me a (story|joke)\b',
        r'\bcreate (a|an)\b',
        r'\bmake up\b',
        r'\bimagine\b.*\bstory\b',
    ]
    
    for pattern in creative_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question requires creative generation with infinite valid responses',
                'epistemic_max': False,
                'aleatoric_max': True,
                'category': 'MAXIMUM VARIABILITY - Creative Task',
                'explanation': 'This is a creative task with infinite equally valid outputs. '
                              'Aleatoric uncertainty is 100% - there is no single "correct" answer, '
                              'and every generation will be different by design.'
            }
    
    # 2. Subjective opinions/preferences
    opinion_patterns = [
        r'\b(what do you think|your opinion|how do you feel)\b',
        r'\bshould i\b',
        r'\bis .* better than\b',
        r'\bfavorite\b',
        r'\bprefer\b',
        r'\bbest\b.*\b(for me|to)\b',
    ]
    
    for pattern in opinion_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question is subjective/opinion-based with no objectively correct answer',
                'epistemic_max': False,
                'aleatoric_max': True,
                'category': 'MAXIMUM VARIABILITY - Subjective Opinion',
                'explanation': 'This question asks for subjective opinions or preferences. '
                              'Aleatoric uncertainty is 100% - there is no objectively correct answer, '
                              'only personal preferences and opinions that vary infinitely.'
            }
    
    # 3. Random number/choice generation
    random_patterns = [
        r'\brandom (number|choice)\b',
        r'\bpick .* for me\b',
        r'\bchoose .* for me\b',
        r'\bgenerate a random\b',
    ]
    
    for pattern in random_patterns:
        if re.search(pattern, question_lower):
            return {
                'should_run': False,
                'reason': 'Question asks for random/arbitrary selection',
                'epistemic_max': False,
                'aleatoric_max': True,
                'category': 'MAXIMUM VARIABILITY - Random Selection',
                'explanation': 'This question requests a random or arbitrary choice. '
                              'Aleatoric uncertainty is 100% - by definition, there is no "correct" answer, '
                              'and the result should vary each time.'
            }
    
    # Question appears answerable - proceed with normal processing
    return {
        'should_run': True,
        'reason': 'Question appears to be factual and answerable',
        'epistemic_max': False,
        'aleatoric_max': False,
        'category': 'ANSWERABLE',
        'explanation': None
    }

def categorize_uncertainty_types(llm, question, answer, temperature=0.3):
    """
    Use LLM to categorize the specific types of epistemic and aleatoric uncertainty
    present in the question and answer.
    
    Returns dict with epistemic_categories and aleatoric_categories
    """
    
    categorization_prompt = f"""Analyze the following question and answer to identify specific types of uncertainty.

Question: {question}
Answer: {answer}

Categorize the uncertainty into specific types:

EPISTEMIC UNCERTAINTY (Knowledge Gaps - what the model doesn't know):
1. Temporal - Information that changes over time (outdated training data)
2. Domain-specific - Specialized knowledge outside training
3. Factual gaps - Missing facts or data in training
4. Contextual - Lacking specific context needed to answer
5. Ambiguous terminology - Unclear terms or definitions

ALEATORIC UNCERTAINTY (Inherent Variability - multiple valid answers):
1. Linguistic ambiguity - Question can be interpreted multiple ways
2. Under-specification - Question lacks necessary details
3. Subjective - No objectively correct answer
4. Creative freedom - Multiple equally valid creative responses
5. Probabilistic - Answer depends on random/unpredictable factors

For EACH uncertainty type present, rate it as: NONE, LOW, MEDIUM, or HIGH

Respond in this exact format:
EPISTEMIC:
- Temporal: [NONE/LOW/MEDIUM/HIGH]
- Domain-specific: [NONE/LOW/MEDIUM/HIGH]
- Factual gaps: [NONE/LOW/MEDIUM/HIGH]
- Contextual: [NONE/LOW/MEDIUM/HIGH]
- Ambiguous terminology: [NONE/LOW/MEDIUM/HIGH]

ALEATORIC:
- Linguistic ambiguity: [NONE/LOW/MEDIUM/HIGH]
- Under-specification: [NONE/LOW/MEDIUM/HIGH]
- Subjective: [NONE/LOW/MEDIUM/HIGH]
- Creative freedom: [NONE/LOW/MEDIUM/HIGH]
- Probabilistic: [NONE/LOW/MEDIUM/HIGH]

Your response:"""
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert in uncertainty quantification and statistical analysis. Be precise and objective."},
            {"role": "user", "content": categorization_prompt}
        ],
        max_tokens=200,  # Reduced from 300
        temperature=temperature
    )
    
    categorization_text = response['choices'][0]['message']['content'].strip()
    
    # Parse the response
    epistemic_categories = {}
    aleatoric_categories = {}
    
    lines = categorization_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if 'EPISTEMIC:' in line:
            current_section = 'epistemic'
        elif 'ALEATORIC:' in line:
            current_section = 'aleatoric'
        elif line.startswith('-') and ':' in line:
            # Parse category line
            parts = line.split(':', 1)
            category = parts[0].replace('-', '').strip()
            level = parts[1].strip().upper()
            
            # Extract just the level (NONE, LOW, MEDIUM, HIGH)
            for lvl in ['NONE', 'LOW', 'MEDIUM', 'HIGH']:
                if lvl in level:
                    level = lvl
                    break
            
            if current_section == 'epistemic':
                epistemic_categories[category] = level
            elif current_section == 'aleatoric':
                aleatoric_categories[category] = level
    
    return {
        'epistemic_categories': epistemic_categories,
        'aleatoric_categories': aleatoric_categories,
        'raw_analysis': categorization_text
    }

def generate_clarifying_questions(llm, question, answer, uncertainty_categories, temperature=0.5):
    """
    Generate clarifying questions based on detected uncertainties to help improve answer quality.
    
    Returns list of clarifying questions
    """
    
    # Check if there are significant uncertainties that warrant clarification
    epistemic = uncertainty_categories['epistemic_categories']
    aleatoric = uncertainty_categories['aleatoric_categories']
    
    # Determine which uncertainties are medium or high
    high_uncertainties = []
    for category, level in epistemic.items():
        if level in ['MEDIUM', 'HIGH']:
            high_uncertainties.append(f"Epistemic: {category} ({level})")
    for category, level in aleatoric.items():
        if level in ['MEDIUM', 'HIGH']:
            high_uncertainties.append(f"Aleatoric: {category} ({level})")
    
    # If no significant uncertainties, no questions needed
    if not high_uncertainties:
        return []
    
    clarification_prompt = f"""Given the following question and answer, along with identified uncertainties, generate 2-4 clarifying questions that would help provide a more accurate and complete answer.

Original Question: {question}
Answer Provided: {answer}

Detected Uncertainties:
{chr(10).join('- ' + u for u in high_uncertainties)}

Generate clarifying questions that would:
1. Resolve ambiguities in the original question
2. Gather missing context or details
3. Specify temporal scope (when applicable)
4. Narrow down interpretation options
5. Identify the specific information needed

Provide ONLY the questions, one per line, numbered. Keep them concise and specific.

Your clarifying questions:"""
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates precise clarifying questions to improve answer quality."},
            {"role": "user", "content": clarification_prompt}
        ],
        max_tokens=100,  # Reduced from 200
        temperature=temperature
    )
    
    questions_text = response['choices'][0]['message']['content'].strip()
    
    # Parse questions (each line is a question)
    questions = []
    for line in questions_text.split('\n'):
        line = line.strip()
        # Remove numbering if present
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if line and len(line) > 10:  # Valid question
            questions.append(line)
    
    return questions

def calculate_entropy(probs):
    """Calculate Shannon entropy from probabilities"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))

def grade_component(value, low_threshold, high_threshold, reverse=False):
    """
    Grade a component on a scale of A-F
    reverse=True means lower values are better
    """
    if reverse:
        value = -value
        low_threshold, high_threshold = -high_threshold, -low_threshold
    
    if value <= low_threshold:
        return 'A', 'Excellent'
    elif value <= low_threshold + (high_threshold - low_threshold) * 0.25:
        return 'B', 'Good'
    elif value <= low_threshold + (high_threshold - low_threshold) * 0.5:
        return 'C', 'Fair'
    elif value <= low_threshold + (high_threshold - low_threshold) * 0.75:
        return 'D', 'Poor'
    else:
        return 'F', 'Critical'

def check_knowledge_availability(llm, question, temperature=0.3):
    """
    Check if the model has sufficient background knowledge to answer the question.
    
    This is done by asking the model to self-assess its knowledge about the topic
    and analyzing the confidence in its self-assessment.
    
    Returns:
        dict with 'has_knowledge' (bool), 'confidence' (float), 'explanation' (str)
    """
    
    knowledge_check_prompt = f"""Given the following question, do you have sufficient factual knowledge and context to provide an accurate answer? 

Question: {question}

Respond with ONLY one of these options:
1. "YES - I have sufficient knowledge to answer this accurately"
2. "PARTIAL - I have some knowledge but may lack recent or complete information"  
3. "NO - I do not have sufficient knowledge to answer this accurately"

Your response:"""
    
    # Get model's self-assessment
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an honest AI assistant. Accurately assess your own knowledge limitations."},
            {"role": "user", "content": knowledge_check_prompt}
        ],
        max_tokens=20,  # Reduced from 50
        temperature=temperature,
        logprobs=True,
        top_logprobs=5
    )
    
    assessment = response['choices'][0]['message']['content'].strip()
    
    # Analyze confidence in self-assessment
    logprobs_data = response['choices'][0].get('logprobs', {})
    if logprobs_data and 'content' in logprobs_data:
        # Get probabilities of first few tokens (most important for YES/PARTIAL/NO)
        first_token_probs = []
        for i, token_info in enumerate(logprobs_data['content'][:5]):  # First 5 tokens
            if token_info and 'logprob' in token_info:
                first_token_probs.append(np.exp(token_info['logprob']))
        
        assessment_confidence = np.mean(first_token_probs) if first_token_probs else 0.5
    else:
        assessment_confidence = 0.5
    
    # Determine knowledge status
    has_knowledge = True
    knowledge_level = "full"
    explanation = ""
    
    assessment_upper = assessment.upper()
    
    # Check in priority order (YES first, then PARTIAL, then NO)
    if "YES" in assessment_upper and assessment_upper.find("YES") < 30:
        has_knowledge = True
        knowledge_level = "full"
        explanation = "Model indicates it has sufficient knowledge to answer"
    elif "PARTIAL" in assessment_upper:
        has_knowledge = True
        knowledge_level = "partial"
        explanation = "Model has partial knowledge but may lack complete or recent information"
    elif assessment_upper.startswith("NO") or (assessment_upper.find(" NO ") >= 0 and assessment_upper.find(" NO ") < 10):
        has_knowledge = False
        knowledge_level = "insufficient"
        explanation = "Model indicates it lacks sufficient knowledge to answer this question"
    else:
        # Uncertain response
        has_knowledge = True
        knowledge_level = "uncertain"
        explanation = "Model's self-assessment was unclear"
    
    return {
        'has_knowledge': has_knowledge,
        'knowledge_level': knowledge_level,
        'confidence': assessment_confidence,
        'assessment': assessment,
        'explanation': explanation
    }

def explain_uncertainty_causes(uncertainty_analysis, token_probs, knowledge_check=None):
    """Generate detailed explanations for why uncertainty is at current levels"""
    
    explanations = []
    
    # Add knowledge availability as first factor if available
    if knowledge_check:
        if knowledge_check['knowledge_level'] == 'insufficient':
            explanations.append("⚠ CRITICAL: Model lacks background knowledge for this question")
        elif knowledge_check['knowledge_level'] == 'partial':
            explanations.append("⚠ Model has only partial knowledge - answer may be incomplete")
        elif knowledge_check['knowledge_level'] == 'full':
            if knowledge_check['confidence'] > 0.7:
                explanations.append("✓ Model has strong background knowledge for this topic")
            else:
                explanations.append("~ Model claims knowledge but with low confidence")
        else:
            explanations.append("~ Model's knowledge level is unclear")
    
    # Analyze aleatoric uncertainty
    aleatoric = uncertainty_analysis['aleatoric_mean']
    if aleatoric < 0.5:
        explanations.append("✓ Low aleatoric uncertainty suggests the question has a clear, definitive answer")
    elif aleatoric < 1.5:
        explanations.append("~ Moderate aleatoric uncertainty indicates some inherent ambiguity in the question")
    else:
        explanations.append("⚠ High aleatoric uncertainty shows multiple valid interpretations or creative freedom")
    
    # Analyze epistemic uncertainty
    epistemic = uncertainty_analysis['epistemic_mean']
    if epistemic < 0.01:
        explanations.append("✓ Low epistemic uncertainty indicates strong model knowledge")
    elif epistemic < 0.05:
        explanations.append("~ Moderate epistemic uncertainty suggests some knowledge gaps")
    else:
        explanations.append("⚠ High epistemic uncertainty reveals significant model uncertainty")
    
    # Analyze token consistency
    token_variance = np.var(token_probs)
    if token_variance < 0.01:
        explanations.append("✓ Consistent token probabilities throughout the response")
    elif token_variance < 0.05:
        explanations.append("~ Some variation in token confidence across the response")
    else:
        explanations.append("⚠ High variation suggests uncertainty concentrated in specific parts")
    
    # Check for low probability tokens
    low_prob_tokens = [p for p in token_probs if p < 0.3]
    if len(low_prob_tokens) == 0:
        explanations.append("✓ All tokens have strong probability (>30%)")
    elif len(low_prob_tokens) < len(token_probs) * 0.2:
        explanations.append(f"~ {len(low_prob_tokens)} tokens with weak probability (<30%)")
    else:
        explanations.append(f"⚠ {len(low_prob_tokens)} tokens with weak probability - model struggling")
    
    return explanations

def analyze_uncertainty(logprobs_data, num_samples=5):
    """
    Analyze aleatoric and epistemic uncertainty from logprobs
    
    Aleatoric Uncertainty: Inherent randomness in the data/task
        - Measured by entropy of token probability distributions
        - High when model sees multiple valid options (e.g., creative tasks)
    
    Epistemic Uncertainty: Model's lack of knowledge
        - Measured by variance in top token probabilities
        - High when model is uncertain about which answer is correct
    """
    
    if not logprobs_data or 'content' not in logprobs_data:
        return None
    
    aleatoric_scores = []  # Entropy per token
    epistemic_scores = []  # Probability spread per token
    token_details = []
    prob_gaps = []  # Track probability gaps (confidence margins)
    
    for token_info in logprobs_data['content']:
        if not token_info or 'token' not in token_info:
            continue
            
        token = token_info['token']
        
        # Get top alternative tokens and their probabilities
        top_logprobs = token_info.get('top_logprobs', [])
        
        if top_logprobs:
            # Extract probabilities from top alternatives
            probs = [np.exp(item['logprob']) for item in top_logprobs]
            
            # Aleatoric: entropy of the distribution (inherent randomness)
            entropy = calculate_entropy(probs)
            aleatoric_scores.append(entropy)
            
            # Epistemic: variance/spread in top probabilities (knowledge uncertainty)
            # High variance means model is uncertain between multiple options
            prob_variance = np.var(probs)
            epistemic_scores.append(prob_variance)
            
            # Calculate probability gap between top 2 choices
            prob_gap = probs[0] - probs[1] if len(probs) > 1 else probs[0]
            prob_gaps.append(prob_gap)
            
            token_details.append({
                'token': token,
                'top_prob': probs[0],
                'entropy': entropy,
                'variance': prob_variance,
                'prob_gap': prob_gap,
                'alternatives': [(item['token'], np.exp(item['logprob'])) 
                                for item in top_logprobs[:3]]
            })
    
    if not aleatoric_scores or not epistemic_scores:
        return None
    
    return {
        'aleatoric_mean': np.mean(aleatoric_scores),
        'aleatoric_max': np.max(aleatoric_scores),
        'epistemic_mean': np.mean(epistemic_scores),
        'epistemic_max': np.max(epistemic_scores),
        'total_uncertainty': np.mean(aleatoric_scores) + np.mean(epistemic_scores),
        'avg_prob_gap': np.mean(prob_gaps),
        'token_details': token_details
    }

def main():
    print("=== Module 8: Hackathon Interactive Chat with LLM Classification ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False, logits_all=True)

    # Initialize the LLM-based classifier for initial screening
    print("Initializing LLM-based prompt classifier...\n")
    try:
        classifier = LLMClassifier(SCREENING_CATEGORIES, llm)
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize classifier: {e}")
        print("Proceeding without classification...\n")
        classifier = None

    # Default generation parameters (optimized for speed)
    temperature = 0.7
    top_p = 0.95
    max_tokens = 100  # Reduced from 250 for faster generation
    
    # Test prompts - reduced set for faster testing
    test_prompts = [
        # Low uncertainty - simple factual
        "What is the capital of France?",
        
        # Moderate uncertainty - requires knowledge
        "What is the best programming language?",
        
        # Temporal uncertainty - requires current date/time context
        "Who is the president now?",
        
        # High epistemic uncertainty - unknowable
        "What is the weather in Los Angeles today?",
        
        # High aleatoric uncertainty - creative
        "Tell me a story about a dragon",
    ]
    
    print("Running analysis on multiple prompts (ordered by increasing uncertainty)...\n")
    print("="*70)
    
    for idx, prompt in enumerate(test_prompts, 1):
        print("\n" + "#"*70)
        print(f"# TEST {idx}/{len(test_prompts)}")
        print("#"*70)
        
        # Ask the specific question
        print("\n" + "="*70)
        print(f"Question: {prompt}")
        print("="*70 + "\n")

        # STEP 0A: LLM CLASSIFICATION - Screen with LLM classifier first
        if classifier:
            print("🔍 LLM CLASSIFICATION SCREENING\n")
            llm_result = classifier.classify(prompt)
            print(f"Primary Category: {llm_result['category']}")
            print(f"Confidence: {llm_result['confidence']:.3f}")
            print(f"Raw Response: {llm_result['raw_response']}\n")
            
            # Determine if we should proceed based on category
            category = llm_result['category']
            proceed_with_generation = True
            warning_message = None
            
            if category.startswith('risky-'):
                risk_type = category.replace('risky-', '')
                if risk_type in ['temporal', 'future', 'personal']:
                    proceed_with_generation = False
                    warning_message = f"⚠️  High epistemic uncertainty detected via LLM classification ({risk_type})"
                elif risk_type in ['creative', 'opinion', 'random']:
                    warning_message = f"⚠️  High aleatoric uncertainty detected via LLM classification ({risk_type})"
            
            if warning_message:
                print(f"\n{warning_message}")
            
            if not proceed_with_generation:
                print("\n" + "="*70)
                print("⚠️  EXECUTION STOPPED - LLM CLASSIFIER FLAGGED AS RISKY")
                print("="*70 + "\n")
                print("The LLM classifier identified this prompt as requiring:")
                if 'temporal' in category:
                    print("  • Real-time or current information")
                elif 'future' in category:
                    print("  • Prediction of future events")
                elif 'personal' in category:
                    print("  • Private personal information")
                print("\nThese questions have maximum epistemic uncertainty.")
                print("Model output would be fabricated/hallucinated.\n")
                print("="*70)
                continue
            
            print("\n" + "="*70)

        # STEP 0B: PATTERN-BASED UNCERTAINTY CHECK
        uncertainty_category = categorize_question_uncertainty(prompt)
        
        print(f"📋 Question Category: {uncertainty_category['category']}")
        print(f"Proceed: {'✓ YES' if uncertainty_category['should_run'] else '✗ NO'}\n")
        
        if not uncertainty_category['should_run']:
            # Maximum uncertainty detected - stop here
            print("\n" + "="*70)
            print("⚠️  EXECUTION STOPPED - MAXIMUM UNCERTAINTY")
            print("="*70 + "\n")
            
            print(f"{uncertainty_category['explanation']}\n")
            
            if uncertainty_category['epistemic_max']:
                print("🎯 Epistemic Uncertainty: 100% (MAXIMUM)")
                print("   The model fundamentally cannot know this answer")
                print("   Running it would produce hallucinated/fabricated output\n")
                
            if uncertainty_category['aleatoric_max']:
                print("📊 Aleatoric Uncertainty: 100% (MAXIMUM)")
                print("   Infinite valid answers exist (inherent to creative/subjective tasks)")
                print("   Every run produces different but equally valid results\n")
            
            print("="*70)
            print("RECOMMENDATION")
            print("="*70)
            
            if uncertainty_category['epistemic_max']:
                print("\n✗ DO NOT USE MODEL OUTPUT - it would be fabricated")
                print("\n✓ Alternative Approaches:")
                print("   • Use real-time API or database")
                print("   • Provide context/data in prompt")
                print("   • Use RAG (Retrieval-Augmented Generation)")
                
            if uncertainty_category['aleatoric_max']:
                print("\n✓ Appropriate for creative/brainstorming tasks")
                print("✗ Not appropriate for factual answers or benchmarking")
            
            print("\n" + "="*70)
            continue  # Skip to next prompt
        
        # Question is answerable - proceed with normal flow
        # STEP 1: Check if model has background knowledge to answer
        print("🔍 KNOWLEDGE CHECK\n")
        knowledge_check = check_knowledge_availability(llm, prompt)
        
        print(f"Level: {knowledge_check['knowledge_level'].upper()}")
        print(f"Confidence: {knowledge_check['confidence']:.1%}")
        
        # Grade knowledge availability
        level = knowledge_check['knowledge_level']
        conf = knowledge_check['confidence']
        
        if level == 'full' and conf > 0.7:
            knowledge_grade, knowledge_desc = 'A', 'Excellent'
        elif level == 'full':
            knowledge_grade, knowledge_desc = 'B', 'Good'
        elif level == 'partial':
            knowledge_grade, knowledge_desc = 'C', 'Partial'
        elif level == 'insufficient':
            knowledge_grade, knowledge_desc = 'F', 'Insufficient'
        elif level == 'uncertain':
            if conf > 0.8: knowledge_grade, knowledge_desc = 'B', 'Good (high conf)'
            elif conf > 0.6: knowledge_grade, knowledge_desc = 'C', 'Fair (mod conf)'
            else: knowledge_grade, knowledge_desc = 'D', 'Poor (low conf)'
        else:
            knowledge_grade, knowledge_desc = 'D', 'Uncertain'
        
        print(f"Grade: {knowledge_grade} ({knowledge_desc})")
        
        if not knowledge_check['has_knowledge']:
            print("\n⚠️  WARNING: Insufficient knowledge - treat answer with caution\n")
        
        # STEP 2: Generate response with logprobs to get probability information
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70 + "\n")
        
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Provide the most concise answer possible - use the absolute minimum words needed to accurately answer the question. Be direct and brief."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            top_logprobs=5
        )

        # Extract response and probability information
        assistant_response = response['choices'][0]['message']['content']
        
        print(f"Assistant: {assistant_response}\n")
        
        # Calculate average token probability (confidence metric)
        if 'logprobs' in response['choices'][0]:
            logprobs_data = response['choices'][0]['logprobs']
            if logprobs_data and 'content' in logprobs_data:
                token_probs = []
                
                for token_info in logprobs_data['content']:
                    if token_info and 'token' in token_info and 'logprob' in token_info:
                        prob = np.exp(token_info['logprob'])
                        token_probs.append(prob)
                
                if token_probs:
                    avg_prob = np.mean(token_probs)
                    min_prob = np.min(token_probs)
                    max_prob = np.max(token_probs)
                    
                    print("\n" + "="*70)
                    print(f"CONFIDENCE: Avg {avg_prob*100:.1f}% | Min {min_prob*100:.1f}% | Max {max_prob*100:.1f}%")
                    print("="*70)
                
                uncertainty_analysis = analyze_uncertainty(logprobs_data)
                
                if uncertainty_analysis:
                    print("\n📊 UNCERTAINTY ANALYSIS\n")
                    
                    uncertainty_categories = categorize_uncertainty_types(llm, prompt, assistant_response)
                    
                    def show_categories(title, cats):
                        print(f"{title}:")
                        for cat, lvl in cats.items():
                            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}.get(lvl, "⚪")
                            print(f"{icon} {cat:25s} {lvl}")
                    
                    epistemic = uncertainty_categories['epistemic_categories']
                    aleatoric = uncertainty_categories['aleatoric_categories']
                    
                    show_categories("Epistemic (Knowledge)", epistemic)
                    print()
                    show_categories("Aleatoric (Variability)", aleatoric)
                    
                    high_epistemic = [k for k, v in epistemic.items() if v == "HIGH"]
                    high_aleatoric = [k for k, v in aleatoric.items() if v == "HIGH"]
                
                if high_epistemic or high_aleatoric:
                    print("\n⚠️  HIGH UNCERTAINTY:")
                    if high_epistemic: print(f"   Epistemic: {', '.join(high_epistemic)}")
                    if high_aleatoric: print(f"   Aleatoric: {', '.join(high_aleatoric)}")
                
                print("\n❓ CLARIFYING QUESTIONS:\n")
                
                clarifying_questions = generate_clarifying_questions(llm, prompt, assistant_response, uncertainty_categories)
                
                if clarifying_questions:
                    for i, q in enumerate(clarifying_questions, 1):
                        print(f"{i}. {q}")
                else:
                    print("✓ None needed")
                
                print("\n" + "="*70)
                print("CONFIDENCE BREAKDOWN")
                print("="*70)
                
                token_variance = np.var(token_probs)
                aleatoric_grade, aleatoric_desc = grade_component(uncertainty_analysis['aleatoric_mean'], 0.5, 2.0)
                epistemic_grade, epistemic_desc = grade_component(uncertainty_analysis['epistemic_mean'], 0.01, 0.08)
                prob_gap_grade, prob_gap_desc = grade_component(uncertainty_analysis['avg_prob_gap'], 0.3, 0.1, reverse=True)
                avg_prob_grade, avg_prob_desc = grade_component(avg_prob, 0.6, 0.3, reverse=True)
                var_grade, var_desc = grade_component(token_variance, 0.02, 0.1)
                min_grade, min_desc = grade_component(min_prob, 0.3, 0.1, reverse=True)
                
                print(f"\n0️⃣  Knowledge: {knowledge_grade} | {level.upper()} @ {conf:.1%}")
                print(f"1️⃣  Avg Prob: {avg_prob_grade} | {avg_prob:.3f} ({avg_prob*100:.1f}%)")
                print(f"2️⃣  Consistency: {var_grade} | Variance {token_variance:.4f}")
                print(f"3️⃣  Aleatoric: {aleatoric_grade} | {uncertainty_analysis['aleatoric_mean']:.3f}")
                print(f"4️⃣  Epistemic: {epistemic_grade} | {uncertainty_analysis['epistemic_mean']:.4f}")
                print(f"5️⃣  Prob Gap: {prob_gap_grade} | {uncertainty_analysis['avg_prob_gap']:.3f}")
                print(f"6️⃣  Min Prob: {min_grade} | {min_prob:.3f} ({min_prob*100:.1f}%)")
                
                grades_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
                all_grades = [knowledge_grade, avg_prob_grade, var_grade, aleatoric_grade, epistemic_grade, prob_gap_grade, min_grade]
                
                weighted_score = (grades_map[knowledge_grade] * 2 + sum(grades_map[g] for g in all_grades[1:])) / 8.0
                
                thresholds = [(3.5, 'A'), (2.5, 'B'), (1.5, 'C'), (0.5, 'D')]
                overall_grade = next((g for t, g in thresholds if weighted_score >= t), 'F')
                
                print(f"\n🎯 OVERALL: {overall_grade} | Grades: {'-'.join(all_grades)} | Score: {weighted_score:.2f}/4.0 (knowledge 2x)")
                
                print("\n📊 DECOMPOSITION:")
                print(f"   Aleatoric: Avg {uncertainty_analysis['aleatoric_mean']:.3f} | Max {uncertainty_analysis['aleatoric_max']:.3f}")
                print(f"   Epistemic: Avg {uncertainty_analysis['epistemic_mean']:.4f} | Max {uncertainty_analysis['epistemic_max']:.4f}")
                print(f"   Total: {uncertainty_analysis['total_uncertainty']:.3f}")
                
                sorted_tokens = sorted(uncertainty_analysis['token_details'], key=lambda x: x['entropy'], reverse=True)[:3]
                if sorted_tokens:
                    print("\n🔍 TOP UNCERTAIN TOKENS:")
                    for i, d in enumerate(sorted_tokens, 1):
                        alts = ', '.join([f"{repr(t)} {p*100:.0f}%" for t, p in d['alternatives'][:2]])
                        print(f"   {i}. {repr(d['token'])} @ {d['top_prob']*100:.0f}% | Entropy {d['entropy']:.3f} | Alts: {alts}")
                
                reliability_score = (1 - min(uncertainty_analysis['total_uncertainty']/5, 1)) * 100
                
                if level == 'insufficient': reliability_score *= 0.3
                elif level == 'partial': reliability_score *= 0.7
                elif conf < 0.5: reliability_score *= 0.8
                
                status_map = {'A': '✓ HIGH', 'B': '✓ HIGH', 'C': '~ MODERATE', 'D': '⚠ LOW', 'F': '⚠ LOW'}
                print(f"\n{status_map[overall_grade]} RELIABILITY | Grade: {overall_grade} | Score: {reliability_score:.0f}%")
    
    print("\n" + "="*70)
    print("\nNow entering interactive mode...")
    print("\nCommands:")
    print("  - Type 'exit' to quit")
    print("  - Type 'set temp X' to change temperature (e.g., 'set temp 0.5')")
    print("  - Type 'set top_p X' to change top_p (e.g., 'set top_p 0.9')")
    print("  - Type 'settings' to view current parameters")
    print("\n" + "="*70)

    # Initialize conversation with system message
    conversation_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and accurate answers."}
    ]

    # Conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() == "exit":
            print("\n👋 Goodbye! Thanks for chatting!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Handle settings commands
        if user_input.lower() == "settings":
            print(f"\n⚙️  Current Settings:")
            print(f"   - Temperature: {temperature}")
            print(f"   - Top P: {top_p}")
            continue
        
        # Handle temperature adjustment
        if user_input.lower().startswith("set temp "):
            try:
                new_temp = float(user_input.split()[-1])
                if 0 <= new_temp <= 2:
                    temperature = new_temp
                    print(f"\n✓ Temperature set to {temperature}")
                else:
                    print("\n⚠️  Temperature should be between 0 and 2")
            except ValueError:
                print("\n⚠️  Invalid temperature value. Use: set temp 0.7")
            continue
        
        # Handle top_p adjustment
        if user_input.lower().startswith("set top_p "):
            try:
                new_top_p = float(user_input.split()[-1])
                if 0 <= new_top_p <= 1:
                    top_p = new_top_p
                    print(f"\n✓ Top P set to {top_p}")
                else:
                    print("\n⚠️  Top P should be between 0 and 1")
            except ValueError:
                print("\n⚠️  Invalid top_p value. Use: set top_p 0.9")
            continue
        
        # Screen with LLM classifier if available
        if classifier:
            llm_result = classifier.classify(user_input)
            category = llm_result['category']
            
            if category.startswith('risky-'):
                risk_type = category.replace('risky-', '')
                if risk_type in ['temporal', 'future', 'personal']:
                    print(f"\n⚠️  Warning: LLM classifier flagged this as '{risk_type}' (high epistemic uncertainty)")
                    print("The answer may be fabricated or hallucinated.")
                elif risk_type in ['creative', 'opinion', 'random']:
                    print(f"\n💡 Info: LLM classifier flagged this as '{risk_type}' (high aleatoric uncertainty)")
                    print("Multiple valid answers exist - output will vary.")
        
        # Add user message to conversation history
        conversation_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        
        response = llm.create_chat_completion(
            messages=conversation_messages,
            max_tokens=512,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )
        
        # Stream the response
        full_response = ""
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                print(content, end="", flush=True)
                full_response += content
        
        print()  # New line after response
        
        # Add assistant response to conversation history
        conversation_messages.append({
            "role": "assistant",
            "content": full_response
        })

    print("\n✓ Module 8 complete")

if __name__ == "__main__":
    main()
