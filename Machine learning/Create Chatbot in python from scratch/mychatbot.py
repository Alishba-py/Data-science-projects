from tkinter import *
import random

class Chatbot:
    def __init__(self, root):
        self.root = root
        self.root.geometry('700x600+250+10')
        self.root.title('My ChatGPT')
        self.root.bind('<Return>', self.ent_funt)

        # Title
        lbl_title = Label(self.root, text='ChatGPT Developer', font=('times new roman', 20, 'bold'))
        lbl_title.place(x=130, y=10)

        # Main Frame
        main_frame = Frame(self.root, bd=2, relief=RAISED, bg='green')
        main_frame.place(x=0, y=60, width=700, height=400)

        # Text Area with Scrollbar
        self.scroll_y = Scrollbar(main_frame, orient=VERTICAL)
        self.text = Text(main_frame, width=65, height=20, font=('arial', 14), relief=RAISED, yscrollcommand=self.scroll_y.set)
        self.scroll_y.pack(side=RIGHT, fill=Y)
        self.scroll_y.config(command=self.text.yview)
        self.text.pack()

        # Search Label
        lbl_search = Label(self.root, text='Search Here', font=('times new roman', 20, 'bold'))
        lbl_search.place(x=20, y=470)

        # Entry Area
        self.ent = StringVar()
        self.entry = Entry(self.root, textvariable=self.ent, font=('times new roman', 15, 'bold'))
        self.entry.place(x=200, y=470, width=400, height=35)

        # Send Button
        self.btn_send = Button(self.root, command=self.send, text='Send', font=('times new roman', 14, 'bold'), bg='black', fg='white')
        self.btn_send.place(x=200, y=520, width=200, height=30)

        # Clear Button
        self.btn_clr = Button(self.root, command=self.clear, text='Clear', font=('times new roman', 14, 'bold'), bg='black', fg='white')
        self.btn_clr.place(x=410, y=520, width=200, height=30)

    # Functions
    def ent_funt(self, event):
        self.btn_send.invoke()

    def clear(self):
        self.text.delete("1.0", END)
        self.ent.set("")

    def send(self):
        user_input = self.entry.get().strip()
        self.text.insert(END, f"\n\nYou: {user_input}")
        self.ent.set("")

        responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! What can I do for you?",
            "how are you": "I'm just a chatbot, but I'm functioning as expected. Thank you!",
            "richest person": "Top 3 Richest persons in the world are:\n1. Elon Musk\n2. Bernard Arnault\n3. Jeff Bezos",
            "top richest person": "Top 3 Richest persons in the world are:\n1. Elon Musk\n2. Bernard Arnault\n3. Jeff Bezos",
            "what is data science": "Data Science is a growing field with applications in daily life. Its branches include:\n1. Machine Learning\n2. Deep Learning\n3. NLP (Natural Language Processing).",
            "about data science": "Data Science is a growing field with applications in daily life. Its branches include:\n1. Machine Learning\n2. Deep Learning\n3. NLP (Natural Language Processing).",
            "google": "Google is a leading tech company known for its services like Google Search, Gmail, and Google Chrome.",
            "about google": "Google is a leading tech company known for its services like Google Search, Gmail, and Google Chrome."
        }

        # Generate Bot Response
        response = responses.get(user_input.lower(), "I'm sorry, I don't understand. Can you please rephrase?")
        self.text.insert(END, f"\n\nBot: {response}")

# Run the Chatbot
if __name__ == "__main__":
    root = Tk()
    obj = Chatbot(root)
    root.mainloop()
