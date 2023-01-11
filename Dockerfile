#
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

#
WORKDIR /app

#
COPY ./requirement.txt /code/requirement.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

EXPOSE 5000
#
COPY ./app /app/app
#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
