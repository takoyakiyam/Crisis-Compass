import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QScrollArea, QDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the 'Impact' column with lemmatization and stop word removal
def preprocess_text(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = re.sub(r'\W+', ' ', text.lower())
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)

    df['Processed_Impact'] = df['Impact'].apply(clean_text)
    return df

# Train the LDA model using TF-IDF and 7 topics
def train_lda_model(df, n_topics=7):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1, 2), stop_words='english')
    dtm = vectorizer.fit_transform(df['Processed_Impact'])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online',
                                    max_iter=10, learning_decay=0.7, learning_offset=50.0)
    lda.fit(dtm)

    return lda, vectorizer

# Filter by the selected topic using Sl. No
def filter_by_topic(df, selected_topic, topic_distributions):
    df['Topic_Distribution'] = topic_distributions[:, selected_topic]
    df_filtered_by_topic = df[df['Topic_Distribution'] > 0.1]  # Threshold to focus on top topic matches
    return df_filtered_by_topic

# Filter by country from the events within the selected topic
def filter_by_country(df, selected_country):
    return df[df['Country'] == selected_country]

# --- Initial Screen Class ---
class InitialScreen(QWidget):
    def __init__(self, on_continue):
        super().__init__()
        
        self.on_continue = on_continue  
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title label
        title_label = QLabel("Crisis Compass")
        title_label.setContentsMargins(0, 0, 0, 0)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Subtitle label
        subtitle_label = QLabel("Navigate through the world's events.")
        subtitle_label.setFont(QFont("Arial", 12)) 
        layout.addWidget(subtitle_label, alignment=Qt.AlignCenter)

        # Continue button
        continue_button = QPushButton("Click to continue")
        continue_button.setFixedHeight(50)
        continue_button.setStyleSheet("""
            QPushButton {
                border-radius: 10px;
                border: 2px solid #007BFF;  
                background-color: #007BFF;  
                color: white;  /* White text color */
                font-size: 16px;  /* Font size */
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;  
            }
            QPushButton:pressed {
                background-color: #004085;  
            }
        """)
        continue_button.clicked.connect(self.on_continue)  
        layout.addWidget(continue_button, alignment=Qt.AlignCenter)
        continue_button.setFixedHeight(40)
        continue_button.clicked.connect(self.on_continue)  
        layout.addWidget(continue_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setWindowTitle("Crisis Compass")
        self.setGeometry(100, 100, 400, 300)

# --- PyQt5 UI Setup ---
class CrisisCompass(QMainWindow):
    def __init__(self, df, lda, vectorizer, topic_names):
        super().__init__()

        self.df = df
        self.lda = lda
        self.vectorizer = vectorizer
        self.topic_names = topic_names
        self.selected_topic_index = None
        self.selected_country = None

        self.setWindowTitle("Crisis Compas")
        self.setGeometry(100, 100, 600, 400)

        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Add a scrollable area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # Set the layout of the scroll content to stick to the top
        self.scroll_layout.setAlignment(Qt.AlignTop)  
        self.layout.addWidget(self.scroll_area)

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Create and add the About Dataset button
        self.about_button = QPushButton("About the Dataset")
        self.about_button.setFixedHeight(40)
        self.about_button.clicked.connect(self.show_about_dialog)
        button_layout.addWidget(self.about_button)

        # Create and add the back button
        self.back_button = QPushButton("Back")
        self.back_button.setFixedHeight(40)
        self.back_button.clicked.connect(self.handle_back)
        button_layout.addWidget(self.back_button, alignment=Qt.AlignRight)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Show the initial topic selection
        self.display_topics()

    # Method to show the About dialog
    def show_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About the Dataset")
        dialog.setGeometry(150, 150, 400, 200)

        layout = QVBoxLayout(dialog)

        # Create a QLabel for the title with bold styling
        title_label = QLabel("<b>World Important Events - Ancient to Modern</b>")
        title_label.setAlignment(Qt.AlignCenter)  # Center the title
        layout.addWidget(title_label)

        # Add description of the dataset
        description = QLabel("This dataset spans significant historical milestones from ancient times to the modern era, covering diverse global incidents. It provides a comprehensive timeline of events that have shaped the world, offering insights into wars, cultural shifts, technological advancements, and social movements.")
        description.setWordWrap(True)  # Allow text to wrap
        layout.addWidget(description)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)

        dialog.exec_()  # Open the dialog

    # Display Topics as buttons
    def display_topics(self):
        self.clear_layout(self.scroll_layout)
        title = QLabel ("Select a Topic:")
        title.setFont(QFont("Arial", 16))
        self.scroll_layout.addWidget(title)

        for idx, topic in enumerate(self.topic_names):
            btn = self.create_rounded_button(f"Topic {idx + 1}: {topic}")
            btn.clicked.connect(lambda checked, i=idx: self.select_topic(i))
            self.scroll_layout.addWidget(btn)

    # Handle topic selection and display country options
    def select_topic(self, idx):
        self.selected_topic_index = idx

        # Filter the dataset based on the selected topic
        dtm = self.vectorizer.transform(self.df['Processed_Impact'])
        topic_distributions = self.lda.transform(dtm)
        self.df_filtered_by_topic = filter_by_topic(self.df, idx, topic_distributions)

        # Display available countries
        self.display_countries()

    # Display available countries based on topic
    def display_countries(self):
        self.clear_layout(self.scroll_layout)
        title = QLabel("Select a Country:")
        title.setFont(QFont("Arial", 16))
        self.scroll_layout.addWidget(title)

        available_countries = self.df_filtered_by_topic['Country'].unique()

        for country in available_countries:
            btn = self.create_rounded_button(country)
            btn.clicked.connect(lambda checked, c=country: self.select_country(c))
            self.scroll_layout.addWidget(btn)

    # Handle country selection and display results
    def select_country(self, country):
        self.selected_country = country
        self.df_filtered_by_country = filter_by_country(self.df_filtered_by_topic, country)

        # Display the recommended events
        self.display_events()

    def display_event_details(self, event):
        dialog = QDialog(self)
        dialog.setWindowTitle("Event Details")
        dialog.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout(dialog)

        # Create a QLabel for event name with bold and centered styling
        event_name_label = QLabel(f"<b>{event['Name of Incident']}</b>")
        event_name_label.setAlignment(Qt.AlignCenter)
        event_name_label.setStyleSheet("font-size: 18px; margin-bottom: 10px;")
        layout.addWidget(event_name_label)

        # Define the event details in a list of tuples for easier iteration
        details = [
            ("Date:", event['Date']),
            ("Month:", event['Month']),
            ("Year:", event['Year']),
            ("Country:", event['Country']),
            ("Type of Event:", event['Type of Event']),
            ("Place Name:", event['Place Name']),
            ("Impact:", event['Impact']),
            ("Affected Population:", event['Affected Population']),
            ("Important Person/Group Responsible:", event['Important Person/Group Responsible']),
        ]

        # Create QLabel for each detail
        for label_text, value in details:
            label = QLabel(f"<b>{label_text}</b> {value}")
            layout.addWidget(label)

        # Color coding for the outcome
        outcome_color = {
            "positive": QColor("green"),
            "mixed": QColor("orange"),
            "negative": QColor("red")
        }
        
        # Create a QLabel for the outcome with background color
        outcome_display = event['Outcome']
        outcome_label = QLabel(f"<b>Outcome:</b> {outcome_display}")
        
        # Apply background color if the outcome is recognized
        if outcome_display.lower() in outcome_color:
            outcome_label.setStyleSheet(f"background-color: {outcome_color[outcome_display.lower()].name()}; padding: 5px;")

        layout.addWidget(outcome_label)  # Add the outcome label to the layout

        dialog.exec_()  # Open the dialog

    def display_events(self):
        self.clear_layout(self.scroll_layout)
        title = QLabel("Recommended Events:")
        title.setFont(QFont("Arial", 16))
        title.setAlignment(Qt.AlignCenter)  # Center the title
        self.scroll_layout.addWidget(title)

        if self.df_filtered_by_country.empty:
            label = QLabel("No events available for the selected country.")
            label.setAlignment(Qt.AlignCenter)  # Center the label
            self.scroll_layout.addWidget(label)
        else:
            for idx, event in self.df_filtered_by_country.iterrows():
                btn = self.create_rounded_button(f"{idx + 1}. {event['Name of Incident']} ({event['Year']}, {event['Country']}) - {event['Type of Event']}")
                btn.clicked.connect(lambda checked, e=event: self.display_event_details(e))  # Pass the event
                self.scroll_layout.addWidget(btn)

        # Force update the scroll area layout
        self.scroll_layout.update()  # Update layout
        self.scroll_area.verticalScrollBar().setValue(0) 

    # Create rounded buttons for the UI
    def create_rounded_button(self, text):
        btn = QPushButton(text)
        btn.setFixedHeight(50)
        btn.setStyleSheet("""
            QPushButton {
                border-radius: 15px;
                border: 2px solid #5D5D5D;
                background-color: #F0F0F0;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #D0D0D0;
            }
            QPushButton:pressed {
                background-color: #A0A0A0;
            }
        """)
        return btn

    # Helper method to clear the layout before displaying new options
    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def handle_back(self):
        if self.selected_country is not None:
            self.selected_country = None
            self.display_countries()  # This method clears the layout before adding new items
            self.scroll_area.verticalScrollBar().setValue(0)  # Reset scroll position to top
        elif self.selected_topic_index is not None:
            self.selected_topic_index = None
            self.display_topics()  # This method clears the layout before adding new items
            self.scroll_area.verticalScrollBar().setValue(0)  # Reset scroll position to top
            
def main():
    # Load the dataset
    file_path = 'World Important Dates.csv'  # Adjust to your file path
    df = load_data(file_path)

    # Preprocess the 'Impact' column for LDA
    df = preprocess_text(df)

    # Train the LDA model with 7 topics
    lda, vectorizer = train_lda_model(df, n_topics=7)

    # Manually set the new topic names
    topic_names = [
        "Economic Conflicts and Leadership",
        "Education and Establishment in the Roman Empire",
        "Presidential Influence in WWII and Global Affairs",
        "The Transition from British Rule to Independence",
        "Monarchies, Societal Dynamics, and Muslim-Jewish Relations",
        "The British Parliament's Global Initiatives",
        "Cultural Loss and Historical Significance"
    ]

    # Create the PyQt application
    app = QApplication([])

    # Function to show the main application window
    def show_main_window():
        main_window = CrisisCompass(df, lda, vectorizer, topic_names)
        main_window.show()
        initial_screen.close()  # Close the initial screen

    # Initialize the Initial Screen
    initial_screen = InitialScreen(on_continue=show_main_window)
    initial_screen.show()

    # Start the application event loop
    app.exec_()

if __name__ == "__main__":
    main()
