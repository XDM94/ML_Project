# API для модели

Использовалась предобученная [модель](https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru) с HuggingFace, которая отвечает на вопросы в заданном контексте развёртывается на локальной машине; далее к ней можно писать запросы и получать ответы через API.

## Как это запустить?

1. Нужно загрузить зависимости, перечисленные в requirements.txt:

```
pip install -r requirements.txt
```

2. Теперь запускаем само приложение:

```
uvicorn ask_friend:app
```

## Как взаимодействовать с моделью?

Два способа (работают на Linux):

1. Postman

Направляем POST-запрос с сырой JSON-кой, содержащей вопрос к модели

```
{
    "text": "Что ты любишь?"
}

```

Модель даст такой ответ:

```
{
	'score': 0.5831074118614197,
	'start': 268,
	'end': 300,
	'answer': ' пирожные из мастерской Менделя.'
}
```

2. cURL

Собсно также отправляем POST-запрос, только на этот раз при помощи cURL.
```
curl -X 'POST' \
  'http://localhost:8000/answer/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Как дела?"
}'
```

На этот пример модель ответит вот так:

```
{
	'score': 0.08206571638584137,
	'start': 136,
	'end': 163,
	'answer': ' всё стремительно меняется.'
}

```