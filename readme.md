
<div align="center">
<h1>🫀&nbsp;<a href="https://share.streamlit.io/lkarjun/heartdisease-prediction/experiments/app.py">Personal Key Indicator of Heart Disease <br>Application</a></h1>
End-to-End · Continuous Machine Learning · Github CI/CD
</div>

<br>

<div align="center">
  <a href="https://share.streamlit.io/lkarjun/heartdisease-prediction/app.py">
       <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
  &nbsp;
  <img src="https://img.shields.io/github/pipenv/locked/python-version/lkarjun/heartdisease-prediction?style=flat&color=3c6e71">
 &nbsp;
 <img src="https://img.shields.io/github/last-commit/lkarjun/heartdisease-prediction?color=ffffff">
 &nbsp;
</div>


<div align="center">
  <img src="https://img.shields.io/github/pipenv/locked/dependency-version/lkarjun/heartdisease-prediction/mlflow?color=f0ead2">
 &nbsp;
  <img src="https://img.shields.io/github/pipenv/locked/dependency-version/lkarjun/heartdisease-prediction/dvc?color=dde5b6">
  &nbsp;
  <img src="https://img.shields.io/github/pipenv/locked/dependency-version/lkarjun/heartdisease-prediction/streamlit?color=adc178">
 &nbsp;
  <img src="https://img.shields.io/github/pipenv/locked/dependency-version/lkarjun/heartdisease-prediction/scikit-learn?color=a98467">

</div>

---

<div>
  
  <h2 align='center'>Application Working </h2>
  
<b>Personal Key Indicator of Heart Disease Application</b> is an End-to-End Continuous Machine Learning Project. The backbone of this project is ```Dvc, Mlflow, GitHub CI/CD, and CML```. `Dvc` stores the dataset and trained model in Azure blob, and `Mlflow` tracks all the training, logs artifacts, and metrics, and versionize the updated model in the model registry. Whenever a pull request comes into the experiment branch, the `GitHub action` will pull the dataset from the Azure blob and retrain the model, then posts the `CML` report as a pr comment. If the pull request merges into the experiment branch, the `GitHub action` will pull the dataset, `mlflow` track the training, and `Dvc` pushes the latest model to the Azure blob. At the same time, GitHub Bot pushes the latest model meta commits to the GitHub and sends `CML` reports as the latest push comment. Whenever a code change occurs in the experiment branch, `Streamlit Application` fetches and updates the app and the model.
  
<div align='center'>
  <br>
  <img src="https://user-images.githubusercontent.com/58617251/173622523-36797d24-107c-452c-ab9c-ced4ebe1807d.png" width='519' height='356'>
</div>

  <br><br>
  
  https://user-images.githubusercontent.com/58617251/173742774-1b13a6fc-31fd-4658-9189-142e45322e37.mp4
  
  <h4 align='center'>Github Bot pull request comment after running Github Action on a pull request to the Experiment branch </h4><br>
  <table>
    <tr>
      <td><img src="https://user-images.githubusercontent.com/58617251/173734439-03d1403e-8b4f-46c7-8d1d-615c1a0435d9.png"></td>
      <td><img src="https://user-images.githubusercontent.com/58617251/173734435-c149e33a-d39e-41b7-a4b0-ed450949fa69.png"></td>
    </tr>
  </table>
  
  <br><br>
  
  <h4 align='center'>After merging the pull request</h4><br>
  <table>
    <tr>
      <td><img src="https://user-images.githubusercontent.com/58617251/173736914-fd4c0981-2ec0-4b36-87b8-2feb083a3ec3.png"></td>
      <td><img src="https://user-images.githubusercontent.com/58617251/173736917-235613d0-3601-46a8-8cad-5521d03bc885.png"></td>
    </tr>
  </table>

</div>


