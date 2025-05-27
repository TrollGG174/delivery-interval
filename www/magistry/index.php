<!DOCTYPE html>

<html lang="ru">

<head>
    <title>Получение сегментов маршрута</title>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <!--
        Подключаем файл конфигурации с API-ключом

        Укажите свой API-ключ в файле config.php. Тестовый ключ НЕ БУДЕТ работать на других сайтах.
        Получить ключ можно в Кабинете разработчика: https://developer.tech.yandex.ru/keys/
    -->
    <?php
    include('config.php');  // Подключаем файл с константой
    ?>
    <script src="https://api-maps.yandex.ru/2.1/?lang=ru_RU&coordorder=longlat&apikey=<?= YANDEX_API_KEY; ?>"
            type="text/javascript"></script>
    <script src="https://yandex.st/jquery/2.2.3/jquery.min.js" type="text/javascript"></script>
    <script src="router.js" type="text/javascript"></script>
    <link href="https://yandex.st/bootstrap/2.2.2/css/bootstrap.min.css" rel="stylesheet">
    <link href="router.css" rel="stylesheet">
</head>

<body>
<div id="map"></div>
</body>

</html>