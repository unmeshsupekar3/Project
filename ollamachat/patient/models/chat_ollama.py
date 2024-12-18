import ollama
import json

class OllaChat:
    def __init__(self):
        self.model='llama3.2:3b '

    def chat_llama(self,user_input):

        print('model loaded')
        
        prompt = f"""
        You are a medical data extraction assistant. Your task is to analyze free-text patient information and extract specific details. For each input, provide a Python dictionary containing the following fields:

        1. `name` - Full name of the patient.
        2. `gender` - Gender of the patient (e.g., male, female, other).
        3. `age` - Age of the patient (integer in years).
        4. `weight` - Weight of the patient in kilograms (kg).
        5. `height` - Height of the patient in centimeters (cm).
        6. `BMI` - Body Mass Index (BMI) calculated using the formula: BMI = weight (kg) / (height (m))^2.
        7. `chief_medical_complaint` - The primary medical complaint or reason for the patient's visit.

        Ensure the output is a well-formed Python dictionary containing **only** these seven fields. Fill in missing fields with `null` if the information is unavailable in the input text.
        Below is the user_input to be considered to extract the data
        Here is the user input: {user_input}

        """
        # prompt_text = prompt.format(input_text = user_input1 )

        response = ollama.chat(model=self.model, format='json', messages=[
            {
                'role': 'user',
                'content':prompt,
            },
        ])

        result = response['message']['content']
        result_json = json.loads(result)
        return result_json

# print(response['message']['content'])


if __name__ == "__main__":
    user_input1 = """ John R. Whitaker, a 52-year-old male, stands 5'10" (70 inches) tall and weighs 198
        lbs. Mr. Whitaker has a history of hypertension and type 2 diabetes, both diagnosed in his
        mid-40s, and recently began experiencing worsening peripheral neuropathy in his lower
        extremities. He also reports chronic lower back pain, which he attributes to years of heavy
        lifting in his previous occupation as a construction worker. Over the past six months, John
        has developed shortness of breath during mild exertion, prompting concerns about potential
        early-stage congestive heart failure. Additionally, he struggles with obesity-related sleep
        apnea, contributing to fatigue and cognitive fog throughout the day. Despite his conditions,
        Mr. Whitaker maintains a generally positive outlook but admits to inconsistent medication
        adherence and difficulty following a healthy diet."""

    user_input2 = """For the past six months, Emily J. Rivera has been dealing with persistent chest
        tightness and shortness of breath, especially during moderate activity. She works as a
        schoolteacher and describes her symptoms as worsening under stress, which she initially
        dismissed as anxiety. A thorough evaluation revealed mild asthma, along with borderline high
        cholesterol levels. Emily is 41 years old, 5'5" (65 inches) tall, and weighs 172 lbs. She also
        complains of intermittent joint stiffness in her hands, particularly in the mornings, which her
        physician suspects could be early osteoarthritis. Emily’s sedentary lifestyle and inconsistent
        exercise routine have contributed to her struggles with maintaining a healthy weight, though
        she remains committed to improving her overall health with proper guidance and treatment."""

    user_input3 = """ Karen L. Thompson, a 38-year-old female, is 5'4" (64 inches) tall and weighs 162
        lbs. She has a history of irritable bowel syndrome (IBS) and recurrent migraines, both of
        which have intensified over the past year. Karen also experiences chronic fatigue and joint
        pain, leading her physician to investigate possible early-stage rheumatoid arthritis. She
        reports frequent episodes of dizziness and occasional heart palpitations, which have been
        attributed to mild anemia and elevated stress levels. Karen’s symptoms are exacerbated by
        her demanding job as a paralegal, where long hours and poor posture have contributed to
        persistent neck and shoulder tension. Recently, she has begun experiencing intermittent
        insomnia, further impacting her energy levels and overall well-being.
        """
    
    oc=OllaChat()
    result=oc.chat_llama(user_input=user_input1)
    print(result)

    


