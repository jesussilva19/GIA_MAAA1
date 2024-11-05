FROM jupyter/julia-notebook
LABEL author="Enrique Fernández Blanco"


USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get install -y htop &&\
    pip install matplotlib plotly rich seaborn

# ========== Install Julia packages as application user ==========

RUN julia -e "using Pkg; Pkg.add.([ \
	\"IJulia\", \
	\"FileIO\", \
	\"XLSX\", \
	\"JLD2\", \
	\"DelimitedFiles\", \
	\"CSV\", \
	\"Flux\", \
	\"ScikitLearn\", \
    \"ScikitLearnBase\", \
	\"Plots\", \
	\"MAT\", \
	\"Tables\", \
	\"Images\", \
	\"DataFrames\", \
	\"Statistics\", \
	\"StatsPlots\", \
    \"CSV\", \
    \"MLJ\", \
    \"Random\", \
    \"HypothesisTests\", \
    \"DecisionTree\", \
    \"MLJDecisionTreeInterface\", \
    \"NearestNeighborModels\", \
    \"RDatasets\", \
    \"UrlDownload\", \
    \"XGBoost\", \
    \"LightGBM\", \
    \"CatBoost\"

]); Pkg.update;"

# Pre compile
RUN julia -e "using IJulia,\
	                FileIO,\
	                XLSX,\
	                JLD2,\
	                DelimitedFiles,\
	                CSV,\
	                Flux,\
	                ScikitLearn,\
                    ScikitLearnBase,\
	                Plots,\
	                MAT,\
	                Tables,\
	                Images,\
	                DataFrames,\
	                Statistics,\
	                StatsPlots,\
                    CSV,\
                    MLJ,\
                    Random,\
                    HypothesisTests,\
                    DecisionTree,\
                    MLJDecisionTreeInterface,\
                    NearestNeighborModels,\
                    RDatasets,\
                    UrlDownload,\
                    XGBoost,\
                    LightGBM,\
                    CatBoost \
                    ;"

USER $NB_UID
