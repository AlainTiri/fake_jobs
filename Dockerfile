#
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

#
WORKDIR /code

#
COPY ./requirement.txt /code/requirement.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

#
COPY ./app /app/app
EXPOSE 80
#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
