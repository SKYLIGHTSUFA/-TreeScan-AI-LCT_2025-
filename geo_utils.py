import folium


def create_map(lat, lon, tree_info):
    popup_text = f"""
    <b>🌳 Дерево</b><br>
    <b>Вид:</b> {tree_info.get('species', 'Не определен')}<br>
    <b>Здоровье:</b> {tree_info.get('health', 'Не оценено')}<br>
    <b>Возраст:</b> {tree_info.get('age', 'Не оценен')}
    """
    m = folium.Map(location=[lat, lon], zoom_start=16)
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip="🌳 Нажми для информации о дереве",
        icon=folium.Icon(color="green", icon="tree", prefix="fa"),
    ).add_to(m)
    return m
