// В работе используется версия API Яндекс.Карты 2.1, официальная документация:
// https://yandex.ru/dev/jsapi-v2-1/doc/ru/

// Ждем загрузки API Яндекс.Карт и запускаем инициализацию карты
ymaps.ready(init);

function init() {

    // Переменная для подсчета количества кликов на карте
    let clicks = 0;

    // Создание карты с заданными параметрами
    var myMap = new ymaps.Map("map", {
        center: [55.168349, 61.390194], // Центр карты
        zoom: 13,
    }, {
        searchControlProvider: 'yandex#search', // Использование контроллеров Яндекса
        restrictMapArea: [[55.276992, 61.234321], [55.050197, 61.589074]], // Ограничение области карты
    });

    // Убираем стандартную панель управления пробками
    myMap.controls.remove('trafficControl');

    // Создаем и настраиваем контроль трафика
    var trafficControl = new ymaps.control.TrafficControl({ state: {
            providerKey: 'traffic#actual', // Используем актуальные данные о пробках
            trafficShown: true, // Показываем пробки на карте
            infoLayerShown: true, // Отображаем информационный слой
        }});
    // Добавляем контроль трафика на карту
    myMap.controls.add(trafficControl);

    // Переменные для хранения координат точек
    let firstPoint = [];
    let secondPoint = [];

    // Функция ожидания появления текста в элементе трафика для заполнения начальных данных о пробках и времени
    function waitForText(selector, callback) {
        const interval = setInterval(function () {
            const element = $(selector);
            if (element.length && element.text().trim() !== "" && element.text().trim() !== "Пробки") {
                clearInterval(interval);
                callback(element);
            }
        }, 100); // Периодичность проверки каждые 100 мс
    }

    // Заполняем элементы трафика и времени
    waitForText('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text', function(element) {
        if($('.customSelectControl').val() == 0)
            $('.customSelectControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0]);

        if($('.customTimeControl').val() == '')
            $('.customTimeControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2]);

    });

     // Обработчик нажатия клавиши "C" для сброса точек и данных
    $(document).on('keypress',function(e) {
        if(e.which == 99) {
            firstPoint = [];
            secondPoint = [];
            $('.customTimeControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2]);
            $('.customSelectControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0]);
            $('.customControl').hide().html('');
            myMap.geoObjects.removeAll();
        }
    });

    // Обработчик клика по карте для добавления точек и построения маршрута
    myMap.events.add('click', function (e) {
        if (!myMap.balloon.isOpen()) {
            var coords = e.get('coords');
            if(!firstPoint.length){
                // Добавляем первую точку
                firstPoint = [coords[0].toPrecision(6), coords[1].toPrecision(6)];
                myGeoObject = new ymaps.GeoObject({
                    geometry: {
                        type: "Point",
                        coordinates: firstPoint
                    },
                    properties: {
                        iconContent: 'Точка отправления',
                    }
                }, {
                    preset: 'islands#redStretchyIcon',
                });
                myMap.geoObjects.add(myGeoObject);
            }else if(!secondPoint.length){
                // Добавляем вторую точку
                secondPoint = [coords[0].toPrecision(6), coords[1].toPrecision(6)];
            }

            // Если обе точки добавлены, строим маршрут
            if(firstPoint.length && secondPoint.length) {

                let currentTraffic = $('.customSelectControl').val() != 0 ? $('.customSelectControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0];
                let currentTime = $('.customTimeControl').val() != '' ? $('.customTimeControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2];

                // Запрашиваем маршрут от первой до второй точки
                ymaps.route([
                    firstPoint,
                    secondPoint
                ]).then(function (route) {
                    myMap.geoObjects.add(route); // Добавляем маршрут на карту
                    // Зададим содержание иконок начальной и конечной точкам маршрута.
                    // С помощью метода getWayPoints() получаем массив точек маршрута.
                    // Массив транзитных точек маршрута можно получить с помощью метода getViaPoints.
                    var points = route.getWayPoints(),
                        lastPoint = points.getLength() - 1;
                    // Задаем стиль метки - иконки будут красного цвета, и
                    // их изображения будут растягиваться под контент.
                    points.options.set('preset', 'islands#redStretchyIcon');
                    // Задаем контент меток в начальной и конечной точках.
                    points.get(0).properties.set('iconContent', 'Точка отправления');
                    points.get(lastPoint).properties.set('iconContent', 'Точка прибытия');


                    var way,
                        segments,
                        distance = 0;

                    for (var i = 0; i < route.getPaths().getLength(); i++) {
                        way = route.getPaths().get(i);
                        segments = way.getSegments();
                        for (var j = 0; j < segments.length; j++) {
                            distance += segments[j].getLength();
                        }
                    }
                    distance = distance.toFixed(0);

                    // Отправляем запрос на сервер для получения интервалов доставки
                    $.ajax({
                        url: 'http://127.0.0.1:8000/intervals/',         /* Куда пойдет запрос */
                        method: 'get',             /* Метод передачи (post или get) */
                        dataType: 'html',          /* Тип данных в ответе (xml, json, script, html). */
                        data: {traffic: currentTraffic, distance: distance, time: currentTime},     /* Параметры передаваемые в запросе. */
                        success: function(data){   /* функция которая будет выполнена после успешного запроса.  */
                            data = JSON.parse(data);
                            let interval = data[0]
                            interval = interval.toString().replace(',', ' - ');
                            interval = interval == '0' ? 'Нет доступных интервалов на сегодня' : interval;
                            let time = data[1]
                            time = time.toString().replace('minutes', 'минут').replace('hour', 'часов').replace('-', ' - ');


                            let ReverseGeocoder = ymaps.geocode([firstPoint]);
                            ReverseGeocoder.then(
                                function (res) {

                                    let firstName = res.geoObjects.get(0).properties.get('name');
                                    let ReverseGeocoder2 = ymaps.geocode([secondPoint]);
                                    ReverseGeocoder2.then(
                                        function (res) {
                                            let secondName = res.geoObjects.get(0).properties.get('name');

                                            // Отображаем результаты на экране
                                            $('.customControl').show().html(
                                                '<p>Точка отправления: ' + firstName + '</p>' +
                                                '<p>Точка прибытия: ' + secondName + '</p>' +
                                                '<p>Расстояние: ' + distance + ' метров' + '</p>' +
                                                '<p>Трафик: ' + currentTraffic + ' уровень' + '</p>' +
                                                '<p>Время заказа: ' + currentTime + '</p>' +
                                                '<p>Интервал: ' + interval + '</p>' +
                                                '<p>Время доставки: ' + time + '</p>'
                                            );
                                        },
                                        function (err) {
                                            alert('Ошибка получения второй точки');
                                            console.log(err);
                                        }
                                    );
                                },
                                function (err) {
                                    alert('Ошибка получения первой точки');
                                    console.log(err);
                                }
                            );
                        }
                    });

                }, function (error) {
                    alert('Ошибка отправки запроса к API: ' + error.message);
                });
            }
        }
        else {
            myMap.balloon.close(); // Закрываем баллон при повторном клике
        }

        // Обработчик правого клика для сброса второй точки
        myMap.events.add('contextmenu', function (){
            secondPoint = [];
            myMap.geoObjects.removeAll();
            $('.customControl').hide().html('');
            myGeoObject = new ymaps.GeoObject({
                geometry: {
                    type: "Point",
                    coordinates: firstPoint
                },
                properties: {
                    iconContent: 'Точка отправления',
                }
            }, {
                preset: 'islands#redStretchyIcon',
            });
            myMap.geoObjects.add(myGeoObject);
        });
    });

    // Создание пользовательского контрола для отображения подсказок
    CustomControlClass = function (options) {
        CustomControlClass.superclass.constructor.call(this, options);
        this._$content = null;
        this._geocoderDeferred = null;
    };

    ymaps.util.augment(CustomControlClass, ymaps.collection.Item, {
        onAddToMap: function (map) {
            CustomControlClass.superclass.onAddToMap.call(this, map);
            this._lastCenter = null;
            this.getParent().getChildElement(this).then(this._onGetChildElement, this);
        },

        onRemoveFromMap: function (oldMap) {
            this._lastCenter = null;
            if (this._$content) {
                this._$content.remove();
                this._mapEventGroup.removeAll();
            }
            CustomControlClass.superclass.onRemoveFromMap.call(this, oldMap);
        },

        _onGetChildElement: function (parentDomContainer) {
            // Создаем HTML-элемент с текстом.
            this._$content = $('<div class="customControlHelp">' +
                '<p>Для отмены точки прибытия нажмите правую кнопку мыши</p>' +
                '<p>Для очистки всех точек и данных нажмите кнопку "C"</p>' +
                '</div>').appendTo(parentDomContainer);
            this._mapEventGroup = this.getMap().events.group();
        },
    });

    // Добавляем контрол для подсказки
    var customControlHelp = new CustomControlClass();
    myMap.controls.add(customControlHelp, {
        position: {
            bottom: 90,
            left: 50
        },
    });

    ymaps.util.augment(CustomControlClass, ymaps.collection.Item, {
        onAddToMap: function (map) {
            CustomControlClass.superclass.onAddToMap.call(this, map);
            this._lastCenter = null;
            this.getParent().getChildElement(this).then(this._onGetChildElement, this);
        },

        onRemoveFromMap: function (oldMap) {
            this._lastCenter = null;
            if (this._$content) {
                this._$content.remove();
                this._mapEventGroup.removeAll();
            }
            CustomControlClass.superclass.onRemoveFromMap.call(this, oldMap);
        },

        _onGetChildElement: function (parentDomContainer) {
            // Создаем HTML-элемент с текстом.
            this._$content = $('<div class="customControl"></div>').appendTo(parentDomContainer);
            this._$content.hide();
            this._mapEventGroup = this.getMap().events.group();
        },
    });

    // Добавляем контрол для вывода информации о доставке
    var customControl = new CustomControlClass();
    myMap.controls.add(customControl, {
        float: 'none',
        position: {
            bottom: 90,
            right: 50
        },
    });

    ymaps.util.augment(CustomControlClass, ymaps.collection.Item, {
        onAddToMap: function (map) {
            CustomControlClass.superclass.onAddToMap.call(this, map);
            this._lastCenter = null;
            this.getParent().getChildElement(this).then(this._onGetChildElement, this);
        },

        onRemoveFromMap: function (oldMap) {
            this._lastCenter = null;
            if (this._$content) {
                this._$content.remove();
                this._mapEventGroup.removeAll();
            }
            CustomControlClass.superclass.onRemoveFromMap.call(this, oldMap);
        },

        _onGetChildElement: function (parentDomContainer) {
            let element = $("<select class='customSelectControl'><option value='0'>Трафик</option></select>").appendTo(parentDomContainer);
            for (let i = 1; i <= 10; i++) {
                element.append(`<option value='${i}'>${i}</option>`);
            }
            this._$content = element;
        },
    });

    // Добавление селектора выбора трафика
    var customSelectControl = new CustomControlClass();
    myMap.controls.add(customSelectControl, {
        float: 'none',
        position: {
            top: 10,
            left: 400
        }
    });

    ymaps.util.augment(CustomControlClass, ymaps.collection.Item, {
        onAddToMap: function (map) {
            CustomControlClass.superclass.onAddToMap.call(this, map);
            this._lastCenter = null;
            this.getParent().getChildElement(this).then(this._onGetChildElement, this);
        },

        onRemoveFromMap: function (oldMap) {
            this._lastCenter = null;
            if (this._$content) {
                this._$content.remove();
                this._mapEventGroup.removeAll();
            }
            CustomControlClass.superclass.onRemoveFromMap.call(this, oldMap);
        },

        _onGetChildElement: function (parentDomContainer) {
            let element = $("<input type='time' class='customTimeControl' min='06:00' max='23:00' />").appendTo(parentDomContainer);
            this._$content = element;
        },
    });

    // Добавление селектора выбора времени
    var customTimeControl = new CustomControlClass();
    myMap.controls.add(customTimeControl, {
        float: 'none',
        position: {
            top: 10,
            left: 550
        },
    });
}