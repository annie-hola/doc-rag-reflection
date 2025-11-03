import gradio as gr


def retriever_qa(file, query):
    print(file, query)


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Document Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document. Guarantee halluciation-free answer"
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7870, share=True)