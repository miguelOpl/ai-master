# 🔄 Tutorial: Construyendo Flujos de Trabajo con LangGraph y LangChain

Este proyecto sirve como tutorial práctico para aprender a construir aplicaciones usando LangGraph y LangChain. Implementaremos un analizador de ideas de negocio que demuestra los conceptos fundamentales de ambas tecnologías.

## 📚 Conceptos Clave

### LangGraph
- **StateGraph**: El componente central para crear flujos de trabajo basados en estados
- **Nodos**: Funciones que procesan y transforman el estado
- **Edges**: Conexiones que definen el flujo entre nodos
- **Estado**: Objeto Pydantic que mantiene los datos entre nodos

### LangChain
- **Agents**: Componentes autónomos que realizan tareas específicas
- **Tools**: Funciones que los agentes pueden utilizar
- **Messages**: Sistema de mensajería para la comunicación
- **Prompts**: Plantillas para interactuar con LLMs

## 🏗️ Estructura del Proyecto

```python
# Definición del Estado
class State(BaseModel):
    messages: List[BaseMessage]  # Mensajes del proceso
    stage: ProcessingStage       # Etapa actual
    user_request: UserRequest    # Datos validados
    trends: List[RedditTrend]    # Tendencias encontradas
    opportunities: List[BusinessOpportunity]  # Oportunidades identificadas
```

## 🔄 Implementación del Flujo

### 1. Configuración del Grafo
```python
# Inicialización
graph = StateGraph(State)

# Agregar Nodos
graph.add_node("initial", initial_node)
graph.add_node("searching", search_reddit_node)
graph.add_node("analyzing", analyze_trends_node)

# Definir Conexiones
graph.add_edge(START, "initial")
graph.add_conditional_edges(
    "initial",
    condition_function,
    {True: "searching", False: END}
)
```

### 2. Implementación de Nodos
```python
def initial_node(state: State) -> State:
    """Nodo de validación inicial"""
    # Actualizar estado
    state.stage = ProcessingStage.INITIAL
    
    # Procesar con LLM
    validated_request = client.chat.completions.create(
        model="gpt-4",
        response_model=UserRequest,
        messages=[...]
    )
    
    # Actualizar estado
    state.user_request = validated_request
    return state
```

### 3. Agentes y Herramientas
```python
# Definición de Agente
reddit_agent = Agent(
    model=model,
    system_prompt="""...""",
    result_type=List[RedditTrend]
)

# Herramienta del Agente
@reddit_agent.tool_plain()
def search_reddit(query: str, subreddit: str) -> str:
    """Búsqueda en Reddit"""
    # Implementación...
```

## 🔍 Características Avanzadas

### Manejo de Estado
- Uso de Pydantic para validación de datos
- Estado inmutable entre nodos
- Propagación de errores

### Flujos Condicionales
```python
def should_continue_searching(state: State) -> bool:
    """Determina si continuar la búsqueda"""
    return state.stage == ProcessingStage.SEARCHING and len(state.trends) < 2

graph.add_conditional_edges(
    "searching",
    should_continue_searching,
    {True: "searching", False: "analyzing"}
)
```

### Integración con LLMs
```python
# Uso de Instructor para estructurar salidas
client = instructor.patch(OpenAI())
result = client.chat.completions.create(
    model="gpt-4",
    response_model=UserRequest,
    messages=[...]
)
```

## 🚀 Ejecutando el Proyecto

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno:
```env
OPENAI_API_KEY=tu_api_key
REDDIT_CLIENT_ID=tu_client_id
REDDIT_CLIENT_SECRET=tu_client_secret
```

3. Crear una app en Reddit y obtener las credenciales:
    - [Reddit App](https://www.reddit.com/prefs/apps)
    - **Importante-->** La app debe ser de tipo "script"    

4. Ejecutar el análisis:
```python
from langraph.tools import analyze_business_idea
result = analyze_business_idea("Tu idea de negocio aquí")
```


## 📈 Mejores Prácticas

1. **Diseño del Estado**
   - Mantener el estado inmutable
   - Usar tipos estrictos con Pydantic
   - Documentar cada campo

2. **Estructura de Nodos**
   - Funciones puras que solo modifican el estado
   - Manejo de errores consistente
   - Logging claro

3. **Flujo del Grafo**
   - Transiciones claras y documentadas
   - Condiciones simples y verificables
   - Estados finales bien definidos

## 🔗 Referencias y Documentación

- [LangGraph Conceptos Básicos](https://python.langchain.com/docs/langgraph)
  - [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)
  - [Guía básica de LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

- [LangChain Components](https://python.langchain.com/docs/get_started/introduction)


- [Instructor para Estructurar LLMs](https://python.useinstructor.com/) 

- [Pydantic AI](https://ai.pydantic.dev/)