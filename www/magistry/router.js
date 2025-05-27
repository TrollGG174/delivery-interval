// В работе используется версия API Яндекс.Карты 2.1, официальная документация:
// https://yandex.ru/dev/jsapi-v2-1/doc/ru/

// Ждем загрузки API Яндекс.Карт и запускаем инициализацию карты
ymaps.ready(init);

// Метод addNiagaraPoints добавляет точки из maps.geojson на карту и обрабатывает их в зависимости от режима
function addNiagaraPoints(myMap, objectManager) {
    return $.getJSON('maps.geojson').then(function (geoJson) {
        let uniqueItems = {};
        geoJson.features.forEach(function (obj) {
            if (obj.properties.iconCaption) {
                obj.options = {
                    preset: "islands#greenDotIconWithCaption"
                };
            }

            if (obj.properties.balloonContentBody) {
                obj.properties.balloonContentBody
                    .split(/<\/br>|<br\s*\/?>|\n/)
                    .map(line => line.trim().replace(/,$/, ''))
                    .filter(line => line.length > 0)
                    .forEach(line => {
                        let match = line.match(/^(.+?)\s*–\s*(\d+)\s*шт$/);
                        if (match) {
                            myMap.options.itemCollection.push({
                                name: match[1].trim(),
                                count: parseInt(match[2], 10),
                                warehouse: obj.id
                            });
                        }
                    });

                let maxCounts = {};
                // 1. Shops
                // В корзину можно добавить максимальное значение товара на одной точке
                // В этом примере происходит поиск магазина, в котором доступны сразу все товары
                // 2. Warehouses
                // В корзину можно добавить максимальное количество товара по всем точкам города
                // В этом примере происходит формирование маршрута для сбора заказа по точкам
                if (myMap.options.mode == 'Shops') {
                    myMap.options.itemCollection.forEach(item => {
                        if (!maxCounts[item.name] || item.count > maxCounts[item.name].count) {
                            maxCounts[item.name] = {name: item.name, count: item.count};
                        }
                    });
                } else if (myMap.options.mode == 'Warehouses') {
                    myMap.options.itemCollection.forEach(item => {
                        if (typeof maxCounts[item.name] !== 'undefined') {
                            maxCounts[item.name] = {name: item.name, count: maxCounts[item.name].count + item.count};
                        } else {
                            maxCounts[item.name] = {name: item.name, count: item.count};
                        }
                    });
                }
                uniqueItems = Object.values(maxCounts);
            }
        });

        myMap.options.shops = geoJson;
        objectManager.add(geoJson);
        myMap.geoObjects.add(objectManager);
        return uniqueItems;
    });
}

// Метод reserveItem сравнивает точки и товары в корзине для режима Shops
function reserveItem(myMap) {
    // Найти склады, где товара достаточно
    let cartItems = [];
    $('.productInput').each(function (index, el) {
        cartItems.push({
            name: $(el).attr('data-name'),
            count: $(el).val(),
        })
    });

    $.getJSON('maps.geojson').then(function (geoJson) {
        let validWarehouses = [];
        geoJson.features.forEach(function (obj) {
            // Задаём пресет для меток с полем iconCaption.
            if (obj.properties.iconCaption) {
                obj.options = {
                    preset: "islands#greenDotIconWithCaption"
                }
            }

            // Группируем товары по складам
            const warehouses = {};
            validWarehouses = [];

            myMap.options.itemCollection.forEach(({name, count, warehouse}) => {
                if (!warehouses[warehouse]) warehouses[warehouse] = {};
                warehouses[warehouse][name] = (warehouses[warehouse][name] || 0) + count;
            });

            // Проверяем каждый склад на соответствие корзине
            for (let [warehouseId, items] of Object.entries(warehouses)) {
                let allMatch = true;

                for (let cartItem of cartItems) {
                    const cartCount = parseInt(cartItem.count, 10);
                    if (cartCount > 0) {
                        if (!items[cartItem.name] || items[cartItem.name] < cartCount) {
                            allMatch = false;
                            break;
                        }
                    }
                }

                if (allMatch) validWarehouses.push(Number(warehouseId));
            }
        });
        geoJson.features = geoJson.features.filter(obj => {
            return validWarehouses.includes(obj.id);
        });
        // Добавляем описание объектов в формате JSON в менеджер объектов.
        myMap.geoObjects.removeAll();
        let objectManager = new ymaps.ObjectManager();
        objectManager.add(geoJson);
        myMap.options.shops = geoJson;
        myMap.options.secondPoint = "";
        myMap.options.flag = 0;
        // Добавляем объекты на карту.
        myMap.geoObjects.add(objectManager);
    });
}

// Функция getCombinations для получения комбинаций массивов размера k без повтора и учета порядка
// Пример:
// getCombinations([1, 2, 3], 2);
// -> [[1, 2], [1, 3], [2, 3]]
function getCombinations(arr, k) {
    const results = [];
    const combine = (start, path) => {
        if (path.length === k) {
            results.push([...path]);
            return;
        }
        for (let i = start; i < arr.length; i++) {
            path.push(arr[i]);
            combine(i + 1, path);
            path.pop();
        }
    };
    combine(0, []);
    return results;
}

// Метод permute для формирования всех возможных перестановок массива
function permute(array) {
    const result = [];
    if (array.length === 0) return [[]];
    for (let i = 0; i < array.length; i++) {
        const rest = [...array.slice(0, i), ...array.slice(i + 1)];
        for (let perm of permute(rest)) {
            result.push([array[i], ...perm]);
        }
    }
    return result;
}

// Метод PointsMode заполняет необходимые элементы для работы с режимом Points
function PointsMode(myMap) {
    // Переменные для хранения координат точек
    myMap.options.flag = 0;

    // Функция ожидания появления текста в элементе трафика для заполнения начальных данных о пробках и времени
    function waitForText(selector, callback) {
        const interval = setInterval(function () {
            const element = $(selector);
            if (element.length && element.text().trim() !== "" && element.text().trim() !== "Пробки") {
                clearInterval(interval);
                callback(element);
            }
        }, 100);
    }

    // Заполняем элементы трафика и времени
    waitForText('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text', function (element) {
        if ($('.customSelectControl').val() == 0)
            $('.customSelectControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0]);

        if ($('.customTimeControl').val() == '')
            $('.customTimeControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2]);

    });
}

// Метод NiagaraMode заполняет элементы корзины для работы с режимами Shops и Warehouses
function NiagaraMode(myMap) {
    let objectManager = new ymaps.ObjectManager();

    $('.customBasketControl').show();

    addNiagaraPoints(myMap, objectManager).then(result => {
        // Сбор всех товаров для корзины
        let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

        result.forEach(function (obj, key) {
            html += `
            <div style="display: flex; align-items: center; gap: 10px;">
                <p style="margin: 0; min-width: 150px;">${obj.name}</p>
                <input 
                    type="number" 
                    id="product_${key}"
                    min="0" 
                    max="${obj.count}" 
                    value="0"
                    data-name="${obj.name}" 
                    class="productInput"
                    style="width: 30px; padding: 4px;"
                >
            </div>`;
        });

        html += '</div>';
        $('.customBasketControl').html(html);

        if(myMap.options.mode == 'Shops') {
            $('.productInput').on('change', function name(params) {
                reserveItem(myMap);
            });
        }
    });
}

// Метод sendDataToAI отправляет запрос к нейросети
function sendDataToAI(ajaxParams){
    $.ajax({
        url: 'http://127.0.0.1:8000/intervals/',         /* Куда пойдет запрос */
        method: 'get',             /* Метод передачи (post или get) */
        dataType: 'html',          /* Тип данных в ответе (xml, json, script, html). */
        data: {traffic: ajaxParams.traffic, distance: ajaxParams.distance, time: ajaxParams.time},     /* Параметры передаваемые в запросе. */
        success: function (data) {   /* функция которая будет выполнена после успешного запроса.  */
            data = JSON.parse(data);
            let interval = data[0]
            interval = interval.toString().replace(',', ' - ');
            interval = interval == '0' ? 'Нет доступных интервалов на сегодня' : interval;
            let time = data[1]
            time = time.toString().replace('minutes', 'минут').replace('hour', 'часов').replace('-', ' - ');
            let ReverseGeocoder = ymaps.geocode([ajaxParams.startPoint]);
            ReverseGeocoder.then(
                function (res) {
                    let firstName = res.geoObjects.get(0).properties.get('name');
                    let ReverseGeocoder2 = ymaps.geocode([ajaxParams.endPoint]);
                    ReverseGeocoder2.then(
                        function (res) {
                            let secondName = res.geoObjects.get(0).properties.get('name');
                            // Отображаем результаты
                            $('.customControl').show().html(
                                '<p>Точка отправления: ' + firstName + '</p>' +
                                '<p>Точка прибытия: ' + secondName + '</p>' +
                                '<p>Расстояние: ' + ajaxParams.distance + ' метров' + '</p>' +
                                '<p>Трафик: ' + ajaxParams.traffic + ' уровень' + '</p>' +
                                '<p>Время заказа: ' + ajaxParams.time + '</p>' +
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
}

// Метод init отвечает за создание карты, а также обработчиков для нее
function init() {
    // Создание карты с заданными параметрами
    var myMap = new ymaps.Map("map", {
        center: [61.390194, 55.168349], // Центр карты
        zoom: 13,
    }, {
        searchControlProvider: 'yandex#search', // Использование контроллеров Яндекса
        restrictMapArea: [[61.234321, 55.276992], [61.589074, 55.050197]], // Ограничение области карты,
    });

    myMap.options.mode = 'Points';
    myMap.options.firstPoint = "";
    myMap.options.secondPoint = "";
    myMap.options.coords = [];
    myMap.options.flag = 0;

    // Убираем стандартную панель управления пробками
    myMap.controls.remove('trafficControl');

    // Создаем и настраиваем контроль трафика
    var trafficControl = new ymaps.control.TrafficControl({
        state: {
            providerKey: 'traffic#actual', // Используем актуальные данные о пробках
            trafficShown: true, // Показываем пробки на карте
            infoLayerShown: true, // Отображаем информационный слой
        }
    });
    // Добавляем контроль трафика на карту
    myMap.controls.add(trafficControl);

    // Проверка выбранного режима взаимодействия с картой
    $(document).on('click', 'input[type="radio"]', function () {
        $('.customTimeControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2]);
        $('.customSelectControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0]);
        $('.customControl').hide().html('');
        myMap.options.flag = 0;
        myMap.options.firstPoint = "";
        myMap.options.secondPoint = "";
        myMap.options.itemCollection = [];
        myMap.options.shops = {};
        myMap.geoObjects.removeAll();

        /*
        Режим Points – построение маршрута между двумя точками и получение интервала доставки

        Режим Shops – показ точек в зависимости от наличия товара на них,
        по клику строится маршрут до ближайшей подходящей точки и выводится интервал доставки

        Режим Warehouses – после добавления необходимых товаров по клику на карте строится маршрут
        по всем точкам, на которых эти товары в наличии, а также выводится интервал доставки
        */
        if ($(this).attr('id') === 'Points' && $(this).is(':checked')) {
            $('.customBasketControl').hide();
            myMap.options.mode = 'Points';
            PointsMode(myMap);
        } else if ($(this).attr('id') === 'Shops' && $(this).is(':checked')) {
            myMap.options.mode = 'Shops';
            NiagaraMode(myMap);
        } else if ($(this).attr('id') === 'Warehouses' && $(this).is(':checked')) {
            myMap.options.mode = 'Warehouses';
            NiagaraMode(myMap);
        }
    });

    // По умолчанию запускаем режим работы с точками
    PointsMode(myMap);

    // Обработчик нажатия клавиши "C" для сброса точек и данных
    $(document).on('keypress', function (e) {
        if (e.which == 99) {
            myMap.options.firstPoint = "";
            myMap.options.secondPoint = "";
            myMap.options.flag = 0;
            $('.customTimeControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2]);
            $('.customSelectControl').val($('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0]);
            $('.customControl').hide().html('');
            myMap.geoObjects.removeAll();
            if (myMap.options.mode == 'Warehouses' || myMap.options.mode == 'Shops') {
                let objectManager = new ymaps.ObjectManager();
                objectManager.add(myMap.options.shops);
                myMap.geoObjects.add(objectManager);
            }
        }
    });

    // Обработчик клика по карте для добавления точек и построения маршрута
    myMap.events.add('click', function (e) {
        if (!myMap.balloon.isOpen()) {
            // Событие клика по карте при работе в режиме точек
            if (myMap.options.mode == 'Points') {
                myMap.options.coords = e.get('coords');
                if (!myMap.options.firstPoint.length) {
                    // Добавляем первую точку
                    myMap.options.firstPoint = [myMap.options.coords[0].toPrecision(6), myMap.options.coords[1].toPrecision(6)];
                    myGeoObject = new ymaps.GeoObject({
                        geometry: {
                            type: "Point",
                            coordinates: myMap.options.firstPoint
                        },
                        properties: {
                            iconContent: 'Точка отправления',
                        }
                    }, {
                        preset: 'islands#redStretchyIcon',
                    });
                    myMap.geoObjects.add(myGeoObject);
                } else if (!myMap.options.secondPoint.length) {
                    // Добавляем вторую точку
                    myMap.options.secondPoint = [myMap.options.coords[0].toPrecision(6), myMap.options.coords[1].toPrecision(6)];
                }

                // Если обе точки добавлены, строим маршрут
                if (myMap.options.firstPoint.length && myMap.options.secondPoint.length && !myMap.options.flag) {

                    let currentTraffic = $('.customSelectControl').val() != 0 ? $('.customSelectControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0];
                    let currentTime = $('.customTimeControl').val() != '' ? $('.customTimeControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2];

                    myMap.options.flag = 1;
                    // Запрашиваем маршрут от первой до второй точки
                    ymaps.route([
                        myMap.options.firstPoint,
                        myMap.options.secondPoint
                    ]).then(function (route) {
                        myMap.geoObjects.add(route); // Добавляем маршрут на карту
                        // Зададим содержание иконок начальной и конечной точкам маршрута.
                        // С помощью метода getWayPoints() получаем массив точек маршрута.
                        var points = route.getWayPoints(), lastPoint = points.getLength() - 1;
                        points.options.set('preset', 'islands#redStretchyIcon');
                        // Задаем контент меток в начальной и конечной точках.
                        points.get(0).properties.set('iconContent', 'Точка отправления');
                        points.get(lastPoint).properties.set('iconContent', 'Точка прибытия');
                        var way, segments, distance = 0;
                        for (var i = 0; i < route.getPaths().getLength(); i++) {
                            way = route.getPaths().get(i);
                            segments = way.getSegments();
                            for (var j = 0; j < segments.length; j++) {
                                distance += segments[j].getLength();
                            }
                        }
                        distance = distance.toFixed(0);

                        let ajaxParams = {
                            'traffic': currentTraffic,
                            'distance': distance,
                            'time': currentTime,
                            'startPoint': myMap.options.firstPoint,
                            'endPoint': myMap.options.secondPoint
                        };

                        // Отправляем запрос на сервер для получения интервалов доставки
                        sendDataToAI(ajaxParams);

                    }, function (error) {
                        console.error('Ошибка при построении маршрута:', error);
                    });
                }
            }
            // Событие клика по карте при работе в режиме магазинов
            if (myMap.options.mode == 'Shops') {
                // Получаем координаты текущего клика по карте
                myMap.options.coords = e.get('coords');
                // Добавляем проверку, чтобы избежать повторного добавления точек
                if (!myMap.options.secondPoint.length) {
                    myMap.options.secondPoint = [
                        myMap.options.coords[0].toPrecision(6),
                        myMap.options.coords[1].toPrecision(6)
                    ];

                    let closestPoint = null;
                    let minDistance = Infinity;

                    // Загружаем точки магазинов с координатами и свойствами
                    let geoJsonPoints = myMap.options.shops.features.map(f => ({
                        coords: f.geometry.coordinates,
                        properties: f.properties
                    }));

                    // Ищем ближайшую точку от выбранной на карте
                    geoJsonPoints.forEach(point => {
                        const distance = ymaps.coordSystem.geo.getDistance(myMap.options.secondPoint, point.coords);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestPoint = point;
                        }
                    });

                    // Если ближайшая точка была найдена и ранее маршрут не был построен, то добавляем маршрут
                    if (closestPoint && !myMap.options.flag) {
                        ymaps.route([
                            myMap.options.secondPoint,
                            closestPoint.coords
                        ]).then(function (route) {
                            myMap.geoObjects.add(route);
                            myMap.options.flag = 1;

                            let points = route.getWayPoints();
                            let lastPoint = points.getLength() - 1;

                            points.options.set('preset', 'islands#redStretchyIcon');
                            points.get(0).properties.set('iconContent', 'Точка прибытия');
                            points.get(lastPoint).properties.set('iconContent', 'Точка отправления');

                            let distance = 0;
                            for (let i = 0; i < route.getPaths().getLength(); i++) {
                                let way = route.getPaths().get(i);
                                let segments = way.getSegments();
                                for (let j = 0; j < segments.length; j++) {
                                    distance += segments[j].getLength();
                                }
                            }

                            distance = distance.toFixed(0);

                            let currentTraffic = $('.customSelectControl').val() != 0 ? $('.customSelectControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0];
                            let currentTime = $('.customTimeControl').val() != '' ? $('.customTimeControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2];

                            let ajaxParams = {
                                'traffic': currentTraffic,
                                'distance': distance,
                                'time': currentTime,
                                'startPoint': closestPoint.coords,
                                'endPoint': myMap.options.secondPoint
                            };

                            // Отправляем запрос на сервер для получения интервалов доставки
                            sendDataToAI(ajaxParams);

                        }, function (error) {
                            console.error("Ошибка при построении маршрута:", error);
                        });
                    }
                }
            }
            if (myMap.options.mode == 'Warehouses') {
                myMap.options.coords = e.get('coords');
                if (!myMap.options.secondPoint.length) {
                    myMap.options.secondPoint = [
                        myMap.options.coords[0].toPrecision(6),
                        myMap.options.coords[1].toPrecision(6)
                    ];
                    const userCoords = myMap.options.secondPoint;

                    // Создаём словарь, где ключ — идентификатор склада, значение — словарь {название товара: количество}
                    const warehouseMap = {};
                    myMap.options.itemCollection.forEach(({ name, count, warehouse }) => {
                        if (!warehouseMap[warehouse]) warehouseMap[warehouse] = {};
                        warehouseMap[warehouse][name] = (warehouseMap[warehouse][name] || 0) + count;
                    });

                    // Считываем текущую корзину и формируем объект cart {название товара: количество}
                    const cart = {};
                    $('.productInput').each(function () {
                        const name = $(this).attr('data-name');
                        const count = parseInt($(this).val(), 10);
                        cart[name] = (cart[name] || 0) + count;
                    });

                    // Получаем все ID складов
                    const warehouseIds = Object.keys(warehouseMap);
                    // Список комбинаций складов, которые могут покрыть заказ полностью
                    let validCombos = [];

                    // Перебираем от 1 до N складов
                    for (let k = 1; k <= warehouseIds.length; k++) {
                        // Получаем все комбинации складов длины k
                        const combos = getCombinations(warehouseIds, k);
                        // Проходим по всем возможным комбинациям
                        for (const combo of combos) {
                            // Собираем суммарные остатки товаров в этой комбинации
                            let total = {};
                            // Объединяем товары всех складов из combo
                            combo.forEach(id => {
                                const stock = warehouseMap[id];
                                for (const [name, count] of Object.entries(stock)) {
                                    if (!total[name]) total[name] = 0;
                                    total[name] += count;
                                }
                            });

                            // Проверяем хватает ли товаров из этой комбинации для полного покрытия корзины
                            const coversAll = Object.entries(cart).every(([name, count]) => (total[name] || 0) >= count);

                            // Если комбинация покрывает заказ — сохраняем
                            if (coversAll) {
                                validCombos.push(combo);
                            }
                        }
                        // Если минимальное покрытие найдено, выходим из цикла
                        if (validCombos.length) break;
                    }

                    let bestRoute = null;
                    let bestDistance = Infinity;

                    for (const combo of validCombos) {
                        // Получаем координаты складов из комбинации по их ID
                        const points = combo
                            .map(id => ({
                                id,
                                coords: myMap.options.shops.features.find(f => f.id == id)?.geometry.coordinates
                            }))
                            .filter(p => p.coords)
                            .map(p => ({
                                id: p.id,
                                coords: p.coords.map(Number)
                            }));

                        // Находим все возможные маршруты между этими складами (перестановки)
                        const perms = permute(points);

                        for (const path of perms) {
                            const coordsPath = path.map(p => p.coords);
                            // Добавляем в конец точку пользователя
                            coordsPath.push(userCoords.map(Number));
                            // Считаем суммарное расстояние по маршруту
                            const dist = coordsPath.reduce((sum, pt, i) => {
                                if (i === 0) return sum;
                                return sum + ymaps.coordSystem.geo.getDistance(coordsPath[i - 1], pt);
                            }, 0);
                            // Если маршрут короче — сохраняем как лучший
                            if (dist < bestDistance) {
                                bestDistance = dist;
                                bestRoute = coordsPath;
                            }
                        }
                    }

                    // Если кратчайший маршрут найден – строим его на карте
                    if (bestRoute) {
                        ymaps.route(bestRoute).then(function (route) {
                            myMap.geoObjects.add(route);
                            const points = route.getWayPoints();
                            points.options.set('preset', 'islands#redStretchyIcon');
                            points.get(0).properties.set('iconContent', 'Старт');
                            points.get(points.getLength() - 1).properties.set('iconContent', 'Финиш');

                            let currentTraffic = $('.customSelectControl').val() != 0 ? $('.customSelectControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[0];
                            let currentTime = $('.customTimeControl').val() != '' ? $('.customTimeControl').val() : $('.ymaps-2-1-79-float-button_traffic_left.ymaps-2-1-79-_checked .ymaps-2-1-79-float-button-text').text().split(' ')[2];

                            let distance = Math.floor(bestDistance);

                            let ajaxParams = {
                                'traffic': currentTraffic,
                                'distance': distance,
                                'time': currentTime,
                                'startPoint': points.get(0).geometry._coordinates,
                                'endPoint': myMap.options.secondPoint
                            };

                            // Отправляем запрос на сервер для получения интервалов доставки
                            sendDataToAI(ajaxParams);
                        }, function (error) {
                            console.error('Ошибка при построении маршрута:', error);
                        });
                    }
                }
            }
        } else {
            myMap.balloon.close(); // Закрываем баллон при повторном клике
        }

        myMap.events.add('contextmenu', function () {
            if (typeof myMap.options.firstPoint !== 'undefined' && myMap.options.mode === 'Points') {
                myMap.options.secondPoint = "";
                myMap.options.flag = 0;
                myMap.geoObjects.removeAll();
                $('.customControl').hide().html('');
                myGeoObject = new ymaps.GeoObject({
                    geometry: {
                        type: "Point",
                        coordinates: myMap.options.firstPoint
                    },
                    properties: {
                        iconContent: 'Точка отправления',
                    }
                }, {
                    preset: 'islands#redStretchyIcon',
                });
                myMap.geoObjects.add(myGeoObject);
            }
            if (myMap.options.mode === 'Warehouses' || myMap.options.mode === 'Shops') {
                myMap.options.secondPoint = "";
                myMap.options.flag = 0;
                myMap.geoObjects.removeAll();
                let objectManager = new ymaps.ObjectManager();
                objectManager.add(myMap.options.shops);
                myMap.geoObjects.add(objectManager);
                $('.customControl').hide().html('');
            }
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
            // Создаем HTML-элемент с текстом.
            this._$content = $('<div class="customBasketControl"></div>').appendTo(parentDomContainer);
            this._$content.hide();
            this._mapEventGroup = this.getMap().events.group();
        },
    });

    // Добавляем контрол для вывода информации о доставке
    var customBasketControl = new CustomControlClass();
    myMap.controls.add(customBasketControl, {
        position: {
            top: 50,
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
            let element = $("<div class='customModeControl'>" +
                "<input type='radio' id='Points' name='Modes' class='radio' checked style='margin:0px'/> " +
                "<label for=\"Points\" style='margin-bottom:0px; margin-right: 10px; margin-left: 3px'>Точки</label>" +
                "<input type='radio' id='Shops' name='Modes' class='radio' style='margin:0px'/> " +
                "<label for=\"Shops\" style='margin:0px; margin-right: 10px; margin-left: 3px'>Магазины</label>" +
                "<input type='radio' id='Warehouses' name='Modes' class='radio' style='margin:0px'/> " +
                "<label for=\"Warehouses\" style='margin:0px; margin-left: 3px'>Склады</label>" +
                "</div>").appendTo(parentDomContainer);
            this._$content = element;
        },
    });

    // Добавление селектора выбора времени
    var customModeControl = new CustomControlClass();
    myMap.controls.add(customModeControl, {
        float: 'none',
        position: {
            top: 10,
            left: 750
        },
    });
}