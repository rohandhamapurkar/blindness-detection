FROM tensorflow/tensorflow:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

CMD [ "streamlit", "run", "app.py" ]

EXPOSE 8501