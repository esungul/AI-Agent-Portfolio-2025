from transformers import DistilBertForQuestionAnswering,DistilBertTokenizer
import torch 
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

##### QA Function with answer prediction

def qa_chatbot(question,context):

  """
  Take a question and context , returns the predicted answer 
  Args: 
     questions (str): Users' question 
     context (str) : Context Text containing the answer 
  Returns:
     answer (str): Predicted answer
  """
  inputs  = tokenizer(question,context,return_tensors='pt')
  with torch.no_grad():
    outputs = model(**inputs)
  
  ### Extract answer span 
  start_idx = torch.argmax(outputs.start_logits)
  print(outputs.start_logits)
  end_idx = torch.argmax(outputs.end_logits)
  answer_span = inputs.input_ids[0][start_idx:end_idx+1]
  answer = tokenizer.decode(answer_span)
  return answer
  
  ############ Explore inputs and outputs 
  print("Input keys ", inputs.keys())
  print("Input ID,shape:" , inputs['input_ids'].shape)
  print("Decoded input Text: " , tokenizer.decode(inputs['input_ids'][0]))
  print("Output keys: ", outputs.start_logits[0:,:5])

#### Test the chatbot 
if __name__ == '__main__':
  #### Test Case 1
  question = "What is AI?"
  context = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its"
  answer = qa_chatbot(question,context)
  print(answer)

  ### Test Case 2
  question2 = "What are multimodal models?"
  context2 = "Multimodal models process text, images, and other data to perform tasks in 2025."
  print(f"Question: {question2}\nAnswer: {qa_chatbot(question2, context2)}")


