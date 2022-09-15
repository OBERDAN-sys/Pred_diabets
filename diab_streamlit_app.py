import streamlit as st
import numpy as np
import pickle
from PIL import Image
from lightgbm import LGBMClassifier


@st.cache
def get_image(path: str) -> Image:
    image = Image.open(path)
    return image

image = get_image("diab_st1.jpg")
st.image(image, use_column_width=True)
st.write('Os dados para o exemplo a seguir são originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e '
         'Renais e contêm informações sobre mulheres com pelo menos 21 anos de descendência indígena Pima. Este é '
         'um aplicativo para predição de diabetes de amostra e não pode ser usado como substituto de um conselho '
         'médico real.')

# Carregar o classificador
filename = 'pre_lgbm.pkl'
lgbm_classifier = pickle.load(open(filename, 'rb'))


def main():
     with st.form(key='pred-diab-form', clear_on_submit=True):
        col1, col2 = st.columns(2)
        n_gravidez = col1.slider(label='Nro_gravidez', min_value=0, max_value=20)
        glicose = col1.number_input(label='Glicose:')
        pressao_sanguinea = col1.number_input(label='Pressao_sanguinea:')
        espessura_triceps = col1.number_input(label='Espessura_triceps:')
        insulina = col2.number_input(label='Insulina:')
        imc = col2.number_input(label='Indice Masa Corporal:')
        hist_familiar_D = col2.slider("Histórico familiar diabetes", 0.000, 2.420, 0.471, 0.001)
        idade = col2.slider(label='Idade:', min_value=21, max_value=120)

        submit = st.form_submit_button(label='Faça Predição')

        if submit:
            result = (n_gravidez, glicose, pressao_sanguinea, espessura_triceps, insulina, imc, hist_familiar_D,
                        idade)
            # Carregar o classificador
            filename = 'pre_lgbm.pkl'
            load_classifier = pickle.load(open(filename, 'rb'))

            numpy_data = np.asarray(result)
            input_reshaped = numpy_data.reshape(1, -1)
            # prediction = load_classifier.predict(input_reshaped)
            prediction_proba = load_classifier.predict_proba(input_reshaped)[0][1]

            # Escrevendo a saida
            st.subheader('Probabilidade de Diabetes')
            st.write('Você tem uma probabilidade de {:.2f} % para risco de diabetes, consulte um médico especialista!'
                     .format((prediction_proba)*100))


if __name__ == '__main__':
    main()

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)