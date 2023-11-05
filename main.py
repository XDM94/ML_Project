# Установка библиотек, необходимых для работы кода

# pip install transformers
# pip install torch
# pip install tensorflow


# Импорт модуля pipeline
from transformers import pipeline

# Создание классификатора c задачей определения тональности текста на предобученной модели типа BERT (rubert-base-cased-sentiment)
classifier = pipeline("sentiment-analysis",
                      "blanchefort/rubert-base-cased-sentiment")

# Вывод результатов определения тональности текста моделью на положительном высказывании, отрицательном и нейтральном
statements = ['Я обожаю инженерию машинного обучения!', 'Я ненавижу инженерию машинного обучения!',
              'В нашем Университете преподают инженерию машинного обучения.']

results = classifier(statements)
print(results)


# Функция проверки работы модели, на вход подаётся список с опредлелением моделью тональности предложений, введённых пользователем
def check(results):
    # Тут функция получает значение тональности из полученных данных и добавляет в список
    model_tonality = []
    for i in results:
        temp = i.get('label')
        model_tonality.append(temp)

    # Проверка того, что модель правильно определила позитивную, отрицательную и нейтральную тональность и интерпритация результатов
    if model_tonality == ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        print('Модель работает корректно.')
    else:
        print(
            'Модель не справилась с определением тональности текста, попробуйте ввести более явные по тональности предложения.')


# Функция для определения тональности текста, введённого пользователем. На вход получает текст, на выоде выдаёт результат работы модели
def definition(statements):
    results = classifier(statements)
    print(results)
    return (results)


# Основная функция, которая совмещает в себе остальные. При вызове попросит пользователя ввести несколько предложений различной тональности
# в порядке: позитивная, негативная, нейтральная. Затем выделит благодаря definition() только тональность из ответа модели
# и с помощью check() проверит справилась ли модель.

def main():
    print('Введите три предложения: с позитивной, негативной и нейтральной тональностью текста:')

    tonality = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    statements = []
    for i in range(3):
        statements.append(input('Предложение с {} тональностью: '.format(tonality[i])))
        print("Введённое Вами предложение с {} тональностью : ".format(tonality[i]), statements[i])
    results = definition(statements)
    check(results)


main()

