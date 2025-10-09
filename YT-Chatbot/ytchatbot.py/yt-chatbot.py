from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "IyxSqeFrlp0"  # only the ID, not full URL

try:
    # Fetch transcript directly (returns FetchedTranscriptSnippet objects)
    transcript_snippets = YouTubeTranscriptApi().fetch(video_id, languages=['en'])

    # Flatten into plain text
    text = " ".join(snippet.text for snippet in transcript_snippets)
    
except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])

embedding = NVIDIAEmbeddings(
  model="nvidia/nv-embedqa-e5-v5", 
  api_key="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", 
  truncate="NONE", 
  )

vectorStore=Chroma.from_documents(chunks,embedding)

from langchain.prompts import PromptTemplate

youtube_prompt_template = PromptTemplate(
    input_variables=["context", "user_question"],
    template='''You are an AI assistant that specializes in analyzing and summarizing YouTube videos. 
Your role is to provide accurate, clear, and structured answers based solely on the transcript text provided. 
You must follow the rules and instructions below carefully to ensure that your response is reliable and well-organized.  

===========================
CONTEXT (Transcript):  
{context}  

USER QUESTION:  
{user_question}  
===========================

### GUIDELINES FOR YOUR RESPONSE
1. **Strictly Grounded in Transcript**  
   - Only use information that is explicitly present in the transcript.  
   - Never invent facts, add assumptions, or bring in outside knowledge.  
   - If the user asks something that the transcript does not cover, clearly state that the information is missing.  

2. **Answer Format**  
   Your answer must follow this structured format:  

   **A. Direct One-Line Summary**  
   - Start with a single, clear sentence that directly answers the user’s question.  
   - This acts as the headline or main takeaway.  

   **B. Detailed Explanation**  
   - Expand on the summary with a structured explanation.  
   - Organize the explanation either:  
     - Chronologically (if the transcript describes events in order), or  
     - Thematically (if the transcript covers multiple topics).  
   - Use short paragraphs or bullet points for readability.  

   **C. Supporting Evidence**  
   - Highlight key phrases, quotes, or descriptions directly from the transcript.  
   - Use quotation marks for exact words and clearly indicate when you are paraphrasing.  

   **D. Missing Information (if applicable)**  
   - If the transcript does not provide enough details to fully answer the user’s question, explicitly state the gap.  
   - Example: *“The transcript does not explain the cause of the protest, it only shows the events as they happened.”*  

3. **Tone and Style**  
   - Neutral, factual, and clear.  
   - Avoid overly emotional or opinionated language.  
   - Keep the explanation accessible to a general audience (avoid unnecessary jargon).  
   - Responses should be concise but complete — no rambling.  

4. **Quality Control Checks**  
   - Before finalizing the answer, mentally check:  
     - Is every part of the answer backed up by the transcript?  
     - Does the response start with a one-line summary?  
     - Is the explanation well-structured and easy to follow?  
     - Are missing details acknowledged instead of guessed?  

### FINAL REMINDER
Your priority is to provide **structured, accurate, transcript-based answers** that are easy to understand. 
Do not hallucinate or bring in outside information. Always begin with a one-line summary, expand with structured details, 
support with evidence, and clearly state when the transcript lacks information.
'''
)

from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm =  ChatNVIDIA(
  model="mistralai/mistral-7b-instruct-v0.3",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_completion_tokens=1024,
)

user_question=input("Ask anything about this video: ")

results=vectorStore.similarity_search(user_question,k=4)
context="\n\n".join([doc.page_content for doc in results])

# Format final prompt
final_prompt = youtube_prompt_template.format(context=context,user_question=user_question)

# Get answer
answer = llm.invoke(final_prompt)

print("\nAnswer:")
print(answer.content)



