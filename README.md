
# Лабораторная работа 1

**Вариант 11:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

**Цель работы**:

Получить навыки разработки CI/CD pipeline для ML моделей с достижением метрик моделей и качества.

**Ход работы**:

1. [x] Создать репозитории модели на GitHub, регулярно проводить commit + push в ветку разработки, важна история коммитов;
2. [x] Провести подготовку данных для набора данных, согласно варианту задания;
3. [x] Разработать ML модель с ЛЮБЫМ классическим алгоритмом классификации, кластеризации, регрессии и т. д.;
4. [x] Конвертировать модель из *.ipynb в .py скрипты;
5. [ ] Покрыть код тестами, используя любой фреймворк/библиотеку;
6. [x] Задействовать DVC;
7. [ ] Использовать Docker для создания docker image.
8. [ ] Наполнить дистрибутив конфигурационными файлами:
- [x] config.ini: гиперпараметры модели;
- [ ] Dockerfile и docker-compose.yml: конфигурация создания контейнера и образа модели;
- [x] requirements.txt: используемые зависимости (библиотеки) и их версии;
- [ ] dev_sec_ops.yml: подписи docker образа, хэш последних 5 коммитов в репозитории модели, степень покрытия тестами;
- [ ] scenario.json: сценарии тестирования запущенного контейнера модели.
9. [ ] Создать CI pipeline (Jenkins, Team City, Circle CI и др.) для сборки docker image и отправки его на DockerHub, сборка должна автоматически стартовать по pull request в основную ветку репозитория модели;
10. [ ] Создать CD pipeline для запуска контейнера и проведения функционального тестирования по сценарию, запуск должен стартовать по требованию или расписанию или как вызов с последнего этапа CI pipeline;
11. [ ] Результаты функционального тестирования и скрипты конфигурации CI/CD pipeline приложить к отчёту.

**Результаты работы**:

1. Отчёт о проделанной работе;
2. Ссылка на репозиторий GitHub;
3. Ссылка на docker image в DockerHub;
4. Актуальный дистрибутив модели в zip архиве.

Обязательно обернуть модель в контейнер (этап CI) и запустить тесты внутри контейнера (этап CD).

**Дополнительно** – настроить веб сервер в отдельном контейнере (Apache/nginx + Flask/Django) для обработки запросов к модели в режиме реального времени.

Выполнение дополнительного условия гарантирует четверть количества баллов за контрольный семинар, достаточного для оценки "отлично".
Таким образом, выполнение дополнительного задания для каждой работы (их 4) даст автоматически оценку "отлично" за курс.
