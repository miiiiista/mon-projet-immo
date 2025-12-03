import streamlit as st
import joblib
import pandas as pd

# 1. Charger le modèle (Le Cerveau)
# On utilise @st.cache_resource pour que le site ne recharge pas le modèle à chaque clic
# Ça rend l'app beaucoup plus rapide.
@st.cache_resource
def load_model():
    return joblib.load('mon_super_modele.pkl')

model = load_model()

# 2. L'Interface (Le Visuel)
st.title("🏡 Estimateur de Prix Immobilier (Californie)")
st.write("Entrez les caractéristiques de la maison pour obtenir une estimation.")

# On divise l'écran en 2 colonnes pour faire joli
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Revenu Médian du quartier (en 10k$)", value=5.0, step=0.1)
    house_age = st.slider("Âge de la maison (années)", 1, 50, 20)
    ave_rooms = st.number_input("Nombre moyen de pièces", value=6.0, step=0.5)
    ave_bedrms = st.number_input("Nombre moyen de chambres", value=1.0, step=0.1)

with col2:
    population = st.number_input("Population du quartier", value=1000, step=100)
    ave_occup = st.number_input("Occupants par maison", value=3.0, step=0.1)
    latitude = st.number_input("Latitude (Ex: 34.0 LA / 37.7 SF)", value=37.7)
    longitude = st.number_input("Longitude (Ex: -118.2 LA / -122.4 SF)", value=-122.4)

# On sort de la colonne (reviens tout à gauche, sans espace au début de la ligne)
st.write("---") # Une petite ligne de séparation esthétique
st.subheader("📍 Localisation du bien")

# On crée les données pour la carte avec les variables que l'utilisateur vient de choisir
map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

# On affiche la carte
st.map(map_data, zoom=10)
# --- FIN DE TON AJOUT ---

# 3. La Prédiction (L'Action)
if st.button("💰 Estimer le Prix"):
    # On doit recréer exactement la même structure que lors de l'entraînement
    # Les noms des colonnes doivent être IDENTIQUES
    features = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    prediction = model.predict(features)
    
    # Le prix est en centaines de milliers de dollars dans le dataset (ex: 2.5 = 250k)
    prix_final = prediction[0] * 100000 
    
    # Prix moyen dans le dataset original (environ 206k)
    prix_moyen_californie = 206855 
    delta = prix_final - prix_moyen_californie

    col_resultat, col_vide = st.columns(2)
    
    with col_resultat:
        st.metric(
            label="Prix Estimé", 
            value=f"{prix_final:,.0f} $", 
            delta=f"{delta:,.0f} $ vs Moyenne",
            delta_color="inverse" # Rouge si cher, Vert si pas cher
        )
    # ... après st.success(...)

    st.subheader("🔍 Comprendre la décision")
    
    # On récupère l'importance de chaque critère (c'est un % calculé par le Random Forest)
    importance = model.feature_importances_
    
    # On crée un tableau propre pour l'affichage
    feature_names = ['Revenu', 'Âge', 'Pièces', 'Chambres', 'Population', 'Occupants', 'Latitude', 'Longitude']
    df_importance = pd.DataFrame({
        'Critère': feature_names,
        'Importance': importance
    }).set_index('Critère')

    # On trie du plus important au moins important
    df_importance = df_importance.sort_values(by='Importance', ascending=False)

    # On affiche le graphique à barres
    st.bar_chart(df_importance)
    # --- AJOUT À LA FIN DU FICHIER app.py ---

st.sidebar.markdown("---")
st.sidebar.header("🧪 Zone Laboratoire")
show_lab = st.sidebar.checkbox("Afficher le mode Expérimental")

if show_lab:
    st.markdown("---")
    st.header("🧪 Laboratoire d'Entraînement")
    st.write("Ici, on entraîne un nouveau modèle en direct pour comprendre l'impact des paramètres.")

    # 1. Chargement des données brutes (pour l'expérience)
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    Y = data.target

    # 2. Les Réglages (Hyperparamètres)
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        n_arbres = st.slider("Nombre d'arbres (n_estimators)", 10, 100, 30)
    with col_param2:
        profondeur = st.slider("Profondeur max (max_depth)", 1, 20, 5)

    # 3. Bouton pour lancer l'entraînement
    if st.button("Lancer l'expérience"):
        # Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Entraînement
        with st.spinner('L\'IA retourne à l\'école...'):
            lab_model = RandomForestRegressor(n_estimators=n_arbres, max_depth=profondeur)
            lab_model.fit(X_train, Y_train)
            score = lab_model.score(X_test, Y_test) # Le R² (1.0 est parfait, 0 est nul)
        
        st.success(f"Score de précision (R²) : {score:.2f}")

        # 4. Le Graphique de Vérité (Réalité vs Prédiction)
        preds = lab_model.predict(X_test)
        
        fig, ax = plt.subplots()
        ax.scatter(Y_test, preds, alpha=0.5, color='blue', s=5)
        ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2) # La ligne parfaite
        ax.set_xlabel('Vrai Prix')
        ax.set_ylabel('Prix Prédit')
        ax.set_title('Si les points sont sur la ligne rouge, c\'est parfait.')
        
        st.pyplot(fig)
        
        st.write("""
        **Comment lire ce graphique ?**
        - **Axe X** : Le prix réel de la maison.
        - **Axe Y** : Le prix deviné par l'IA.
        - **Ligne Rouge** : La perfection.
        - **Nuage de points** : Si le nuage est compact autour de la ligne, le modèle est bon. S'il est dispersé, le modèle hésite.
        """)

