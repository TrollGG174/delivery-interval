# Система интервалов доставки заказа

## Инструменты
- PHP
- Apache
- YandexAPI
- Python
- FastAPI
- Torch


## Установка

1) Скачайте docker с официального сайта https://www.docker.com/products/docker-desktop/
2) Получите ключ API для работы с Яндекс Картой https://developer.tech.yandex.ru/keys/
3) Скачайте этот репозиторий на свой компьютер
```shell
   git clone https://github.com/TrollGG174/delivery-interval.git
```
4) Сконфигурируйте файл .env при необходимости
5) Укажите ключ Яндекс API в файле /www/magistry/router.html вместо ###
6) Откройте папку с репозиторием через терминал
7) Отредактируйте файл hosts и добавьте туда запись
```shell
   127.0.0.1 magistry
```
   Подробнее можно прочитать здесь:

   https://help.reg.ru/support/dns-servery-i-nastroyka-zony/rabota-s-dns-serverami/fayl-hosts-gde-nakhoditsya-i-kak-yego-izmenit#2

8) Запустите команду docker-compose up --build
```shell
   docker-compose up --build
```
9) Если все прошло успешно, то сайт будет доступен по следующему адресу http://magistry:8081/

## Описание
- При первом нажатии на карту задается точка отправления
- При втором нажатии задается точка прибытия и строется маршрут
- После построения маршрута отправляется запрос к API нейросети на python для получения интервала и примерного времени доставки

## Управление
- В полях сверху можно задавать произвольное время и трафиик на дороге
- При нажатии правой кнопки мыши точка прибытия удаляется и можно задать другую
- При нажатии на кнопку 'C' убираются обе точки и устанавливаются данные по умолчанию (текущее время и трафик)
