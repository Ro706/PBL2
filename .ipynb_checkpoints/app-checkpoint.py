import tkinter as tk
from tkinter import filedialog
# Function to load and predict an image
def predict_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the image in the GUI
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        # Preprocess the image for the model
        img_for_model = Image.open(file_path).resize((64, 64))
        img_array = np.array(img_for_model) / 255.0  # Rescale like during training
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)[0][0]
        result = "Wildfire" if prediction > 0.5 else "No Wildfire"
        result_label.config(text="Prediction: " + result)

# Setting up the GUI window
root = tk.Tk()
root.title("Forest Fire Detection")
root.geometry("400x400")

# Add widgets
btn = tk.Button(root, text="Upload Image", command=predict_image)
btn.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=20)

root.mainloop()