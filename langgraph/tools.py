from typing import Annotated

from enum import Enum
from typing import Dict, List, Optional

# Pydantic AI para definir los modelos de datos
from pydantic import BaseModel, Field

# Pydantic AI para definir los agentes
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel

# Langchain Reddit Search
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.tools.reddit_search.tool import RedditSearchSchema

# Langchain Core Messages para guardar el historial de mensajes
from langchain_core.messages import BaseMessage, HumanMessage

# Instructor
import instructor

# OpenAI
from openai import OpenAI

# Langgraph
from langgraph.graph import StateGraph, START, END

# Nest Asyncio para sincronizar la ejecución de los agentes
import nest_asyncio
nest_asyncio.apply()

# Para cargar las variables de entorno
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with instructor
base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = instructor.patch(base_client)

class UserRequest(BaseModel):
    """Model to validate if a user request is related to business idea research"""
    is_business_research: bool = Field(
        description="Whether the request is related to business idea research"
    )
    request_description: str = Field(
        description="A cleaned and structured version of the user's request"
    )
    target_market: Optional[str] = Field(
        default=None,
        description="The target market or industry mentioned in the request"
    )
    key_objectives: List[str] = Field(
        default_factory=list,
        description="Main objectives or questions to be researched"
    )

# ================================================
# ESTADO
# ================================================

class ProcessingStage(str, Enum):
    INITIAL = "initial"
    SEARCHING = "searching_reddit"
    ANALYZING = "analyzing_trends"
    PLANNING = "creating_plans"
    COMPLETE = "complete"
    ERROR = "error"
    END="end"

class RedditTrend(BaseModel):
    title: str = Field(description="The post title")
    url: str = Field(description="The post url")
    subreddit: str = Field(description="The subreddit where the post was published")
    content: str = Field(description="The content of the post")
    sentiment_score: Optional[float] = Field(description="The sentiment score of the post")

class BusinessOpportunity(BaseModel):
    trend: RedditTrend
    market_size: str
    target_audience: str
    potential_revenue: str
    risk_level: str
    competitive_analysis: str

class BusinessPlan(BaseModel):
    opportunity: BusinessOpportunity
    executive_summary: str
    implementation_steps: List[str]
    required_resources: Dict[str, str]
    timeline: Dict[str, str]
    estimated_costs: Dict[str, float]

class State(BaseModel):
    """State for Reddit Business Trend Analysis Agent"""
    messages: Annotated[List[BaseMessage], "append"] 
    stage: ProcessingStage = Field(
        default=ProcessingStage.INITIAL,
        description="Current stage of the analysis process"
    )
    
    # User request validation
    user_request: Optional[UserRequest] = Field(
        default=None,
        description="Validated user request information"
    )
    
    # Collection of Reddit trends found
    trends: List[RedditTrend] = Field(
        default_factory=list,
        description="List of trending topics found on Reddit"
    )
    
    # Analyzed business opportunities
    opportunities: List[BusinessOpportunity] = Field(
        default_factory=list,
        description="List of analyzed business opportunities"
    )
    
    # Generated business plans
    plans: List[BusinessPlan] = Field(
        default_factory=list,
        description="List of generated business plans"
    )
    
    # Error tracking
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if any stage fails"
    )
    

# ================================================
# AGENTES
# ================================================

model = OpenAIModel(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

reddit_agent = Agent(
    model=model,
    system_prompt="""
    Eres un experto en buscar tendencias en Reddit especializado en investigar ideas de negocio novedosas
    y que tengan potencial de éxito.
    
    Tu tarea es analizar los resultados de búsqueda de Reddit y convertirlos en tendencias estructuradas.
    Cada tendencia debe tener:
    - Un título claro y descriptivo
    - La URL del post original
    - El subreddit donde se encontró
    - Un resumen del contenido
    - Un análisis de sentimiento (0 a 1, donde 1 es muy positivo)
    
    Asegúrate de que cada tendencia esté bien formada y contenga toda la información necesaria.
    No inventes datos, usa solo la información proporcionada por Reddit.
    """,
    result_type=List[RedditTrend]
)

analyze_trends_agent = Agent(
    model=model,
    system_prompt="""
    Eres un experto en analizar tendencias de Reddit y convertirlas en oportunidades de negocio.
    Para cada tendencia que analices, debes crear una oportunidad de negocio estructurada que incluya:
    
    1. La tendencia original como base
    2. Análisis del mercado:
       - Tamaño estimado del mercado
       - Público objetivo bien definido
       - Potencial de ingresos realista
       - Nivel de riesgo (bajo, medio, alto)
       - Análisis competitivo detallado
    
    Sé específico y realista en tus análisis. Basa tus conclusiones en los datos de la tendencia.
    Responde siempre en español.
    """,
    result_type=List[BusinessOpportunity]
)

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

search = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapper(
        reddit_client_id=client_id,
        reddit_client_secret=client_secret,
        reddit_user_agent=user_agent,
    )
)

# ================================================
# HERRAMIENTAS
# ================================================

@reddit_agent.tool_plain()
def search_reddit(query: str, subreddit: str) -> str:
    """Busca tendencias en Reddit y las estructura"""
    try:
        print(f"🔍 Buscando en r/{subreddit}: {query}")
        
        search_params = RedditSearchSchema(
            query=query,
            sort="top",
            time_filter="month",
            subreddit=subreddit,
            limit="5"
        )
        
        result = search.run(tool_input=search_params.model_dump())
        if not result:
            return "No se encontraron resultados"
            
        return result
        
    except Exception as e:
        print(f"❌ Error en búsqueda de Reddit: {str(e)}")
        return f"Error: {str(e)}"

# ================================================
# NODOS
# ================================================

def initial_node(state: State) -> State:
    """Valida la solicitud del usuario"""
    try:
        # Siempre actualizamos el stage
        state.stage = ProcessingStage.INITIAL
        
        validated_request = client.chat.completions.create(
            model="gpt-4",
            response_model=UserRequest,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """Eres un validador de solicitudes de investigación de ideas de negocio.
                    Analiza la entrada del usuario y determina si está relacionada con la investigación de ideas de negocio o el análisis del mercado.
                    Estructura y valida la solicitud adecuadamente."""
                },
                {
                    "role": "user",
                    "content": f"Por favor, valida esta solicitud:\n\n{state.messages[-1].content}"
                }
            ]
        )
        
        state.user_request = validated_request
        
        if validated_request.is_business_research:
            state.stage = ProcessingStage.SEARCHING
            state.messages.append(HumanMessage(content="✅ Solicitud válida. Iniciando búsqueda..."))
        else:
            state.stage = ProcessingStage.ERROR
            state.error_message = "❌ La petición no está relacionada con investigación de negocios."
            state.messages.append(HumanMessage(content=state.error_message))
            
    except Exception as e:
        state.stage = ProcessingStage.ERROR
        state.error_message = f"Error en validación: {str(e)}"
        state.messages.append(HumanMessage(content=state.error_message))
    
    return state

def search_reddit_node(state: State) -> State:
    """Busca y procesa tendencias de Reddit"""
    try:
        state.messages.append(HumanMessage(content="🔍 Buscando tendencias..."))
        
        # Aseguramos que tenemos una solicitud válida
        if not state.user_request:
            raise ValueError("No hay una solicitud válida para procesar")
            
        # Ejecutamos el agente de Reddit
        result: AgentRunResult[List[RedditTrend]] = reddit_agent.run_sync(
            user_prompt=f"Analiza y estructura las tendencias para: {state.user_request.request_description}"
        )
        trends = result.data
        print("Trends: ", trends)   
        if not trends:
            state.messages.append(HumanMessage(content="⚠️ No se encontraron tendencias relevantes"))
            return state
            
        # Actualizamos el estado
        state.trends = trends
        state.stage = ProcessingStage.ANALYZING
        state.messages.append(HumanMessage(content=f"✅ Se encontraron {len(trends)} tendencias"))
        
    except Exception as e:
        state.stage = ProcessingStage.ERROR
        state.error_message = f"Error en búsqueda: {str(e)}"
        state.messages.append(HumanMessage(content=state.error_message))
    
    return state

def analyze_trends_node(state: State) -> State:
    """Analiza las tendencias y genera oportunidades"""
    try:
        print("Analyze trends node")
        if not state.trends:
            raise ValueError("No hay tendencias para analizar")
            
        state.messages.append(HumanMessage(content="🔍 Analizando tendencias..."))
        
        result: AgentRunResult[List[BusinessOpportunity]] = analyze_trends_agent.run_sync(
            user_prompt=f"Analiza estas tendencias y genera oportunidades de negocio: {state.trends}"
        )
        opportunities = result.data
        print("Opportunities: ", opportunities)
        state.opportunities = opportunities
        state.stage = ProcessingStage.COMPLETE
        state.messages.append(HumanMessage(content=f"✅ Se identificaron {len(opportunities)} oportunidades"))
        
    except Exception as e:
        state.stage = ProcessingStage.ERROR
        state.error_message = f"Error en análisis: {str(e)}"
        state.messages.append(HumanMessage(content=state.error_message))
    
    return state

# ================================================
# MÉTODOS CONDICIONALES
# ================================================

def should_continue_searching(state: State) -> bool:
    """Determina si debemos continuar buscando tendencias"""
    continue_searching = state.stage == ProcessingStage.SEARCHING and (not state.trends or len(state.trends) < 2)
    print("Continue searching: ", continue_searching)
    return continue_searching

def should_analyze_trends(state: State) -> bool:
    """Determina si debemos analizar las tendencias"""
    analyze_trends = state.stage == ProcessingStage.ANALYZING and len(state.trends) >= 2
    print("Analyze trends: ", analyze_trends)
    return analyze_trends

# ================================================
# GRAFO
# ================================================

graph = StateGraph(State)

# Nodos
graph.add_node(ProcessingStage.INITIAL.value, initial_node)
graph.add_node(ProcessingStage.SEARCHING.value, search_reddit_node)
graph.add_node(ProcessingStage.ANALYZING.value, analyze_trends_node)

# Edges
graph.add_edge(START, ProcessingStage.INITIAL.value)

# Conditional edges
graph.add_conditional_edges(
    ProcessingStage.INITIAL.value,
    lambda s: s.user_request and s.user_request.is_business_research,
    {
        True: ProcessingStage.SEARCHING.value,
        False: END
    }
)

graph.add_conditional_edges(
    ProcessingStage.SEARCHING.value,
    should_continue_searching,
    {
        True: ProcessingStage.SEARCHING.value,
        False: ProcessingStage.ANALYZING.value
    }
)


graph.add_edge(ProcessingStage.ANALYZING.value, END)

# Compile
graph = graph.compile()

def analyze_business_idea(user_input: str):
    """Analiza una idea de negocio y muestra los resultados de forma estructurada"""
    # Estado inicial
    initial_state = State(
        messages=[HumanMessage(content=user_input)],
        stage=ProcessingStage.INITIAL
    )
    
    # Ejecutar el grafo
    result_dict = graph.invoke(initial_state)
    
    # Convertir el resultado a un objeto State
    result = State(
        messages=result_dict.get('messages', []),
        stage=result_dict.get('stage', ProcessingStage.ERROR),
        user_request=result_dict.get('user_request'),
        trends=result_dict.get('trends', []),
        opportunities=result_dict.get('opportunities', []),
        plans=result_dict.get('plans', []),
        error_message=result_dict.get('error_message')
    )
    
    # Mostrar resultados
    print("\n🔍 ANÁLISIS DE IDEA DE NEGOCIO")
    print("=" * 50)
    
    # 1. Estado del Proceso
    print(f"\n📊 Estado Final: {result.stage.value}")
    
    if result.error_message:
        print(f"\n❌ Error: {result.error_message}")
        return
    
    # 2. Validación de la Idea
    if result.user_request:
        print("\n🎯 VALIDACIÓN DE LA IDEA")
        print("-" * 30)
        print(f"• Es idea de negocio: {'✅' if result.user_request.is_business_research else '❌'}")
        print(f"• Descripción: {result.user_request.request_description}")
        if result.user_request.target_market:
            print(f"• Mercado objetivo: {result.user_request.target_market}")
        if result.user_request.key_objectives:
            print("\n📋 Objetivos Clave:")
            for i, obj in enumerate(result.user_request.key_objectives, 1):
                print(f"  {i}. {obj}")
    
    # 3. Tendencias Encontradas
    if result.trends:
        print("\n📈 TENDENCIAS EN REDDIT")
        print("-" * 30)
        for i, trend in enumerate(result.trends, 1):
            print(f"\n📌 Tendencia {i}:")
            print(f"• Título: {trend.title}")
            print(f"• Subreddit: r/{trend.subreddit}")
            print(f"• Sentimiento: {'😊' if trend.sentiment_score > 0.5 else '😐' if trend.sentiment_score > 0.3 else '😟'} ({trend.sentiment_score:.2f})")
            print(f"• URL: {trend.url}")
            print(f"• Resumen: {trend.content[:150]}...")
    
    # 4. Oportunidades de Negocio
    if result.opportunities:
        print("\n💡 OPORTUNIDADES DE NEGOCIO")
        print("-" * 30)
        for i, opp in enumerate(result.opportunities, 1):
            print(f"\n🎲 Oportunidad {i}:")
            print(f"• Basada en: {opp.trend.title}")
            print(f"• Tamaño de mercado: {opp.market_size}")
            print(f"• Público objetivo: {opp.target_audience}")
            print(f"• Potencial de ingresos: {opp.potential_revenue}")
            print(f"• Nivel de riesgo: {opp.risk_level}")
            print("\n📊 Análisis competitivo:")
            print(f"{opp.competitive_analysis}")
    
    # 5. Historial de Mensajes
    print("\n📝 HISTORIAL DE PROGRESO")
    print("-" * 30)
    for msg in result.messages:
        print(f"• {msg.content}")

# Para usar en el notebook:
analyze_business_idea("Quiero investigar el mercado de aplicaciones de IA para pequeñas empresas")