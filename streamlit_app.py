
import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Latin Kare & SRG Araştırma Laboratuvarı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık ve İnfo-Graf Üst Alanı
st.title("#306 MUCİT BEYİNLER — Latin Kareleri ve Güçlü Düzenli Graflar Labı")
st.markdown("### KODLARLA TARİHİ İNŞA ETMEK: Kombinatorik Tasarım, Ortogonal Diziler ve Spektral Analiz")
st.markdown("---")

# Yan Menü (Sidebar) - Değişken Parametre Kontrolleri
st.sidebar.header("🎛️ Simülasyon Değişkenleri")
n = st.sidebar.slider("Latin Kare Mertebesi ($n$)", min_value=3, max_value=5, value=4, step=1)
square_mode = st.sidebar.selectbox(
    "MOLS Üretim Stratejisi",
        ["Standart Döngüsel (Cyclic)", "Satır Permütasyonlu (Variant)"]
)
layout_style = st.sidebar.selectbox(
    "Graf Yerleşimi (Layout)",
    ["Spring Layout", "Circular Layout", "Shell Layout"]
)

# 1. Latin Kare ve OA Üretim Motoru
@st.cache_data
def generate_mols_and_graph(n, mode):
    # L1: Standart döngüsel
    L1 = np.array([[(i + j) % n for j in range(n)] for i in range(n)])
    
    # L2: Stratejiye göre ikinci Latin kare
    if mode == "Standart Döngüsel (Cyclic)":
        L2 = np.array([[(i + 2 * j) % n for j in range(n)] for i in range(n)])
    else:
        base_L2 = np.array([[(i + 2 * j) % n for j in range(n)] for i in range(n)])
        perm = list(range(n))
        if n >= 4:
            perm[0], perm[1] = perm[1], perm[0] # Basit permütasyon
        L2 = base_L2[perm, :]

    # Ortogonal Dizi OA(4, n)
    oa = []
    for r in range(n):
        for c in range(n):
            oa.append([r, c, L1[r, c], L2[r, c]])
    oa = np.array(oa)

    # Graf Oluşturma
    v = oa.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(v))

    for i in range(v):
        for j in range(i + 1, v):
            if np.any(oa[i] == oa[j]):
                G.add_edge(i, j)
                
    return L1, L2, oa, G

L1, L2, oa, G = generate_mols_and_graph(n, square_mode)

# 2. Kombinatorik ve Spektral Analiz
degrees = [d for _, d in G.degree()]
is_regular = len(set(degrees)) == 1
d_val = degrees[0] if is_regular else None

# Lambda ve Mu hesaplama
lambdas, mus = set(), set()
nodes = list(G.nodes())
v_nodes = G.number_of_nodes()

for i in range(v_nodes):
    for j in range(i + 1, v_nodes):
        common = len(list(nx.common_neighbors(G, nodes[i], nodes[j])))
        if G.has_edge(nodes[i], nodes[j]):
            lambdas.add(common)
        else:
            mus.add(common)

is_srg = is_regular and len(lambdas) == 1 and len(mus) == 1
lam_val = list(lambdas)[0] if len(lambdas) == 1 else "Değişken"
mu_val = list(mus)[0] if len(mus) == 1 else "Değişken"

# --- Arayüz Yerleşimi (İnfo-Graf Kartları) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Düğüm Sayısı ($v=n^2$)", v_nodes)
col2.metric("Kenar Sayısı", G.number_of_edges())
col3.metric("Derece ($d$)", d_val if is_regular else "Düzenli Değil")
col4.metric("Güçlü Düzenlilik (SRG)", "✅ Sağlanıyor" if is_srg else "❌ Sağlanmıyor")

if is_srg:
    st.success(f"Matematiksel Parametre Seti: $\\text{SRG}(v={v_nodes}, d={d_val}, \\lambda={lam_val}, \\mu={mu_val})$")
else:
    st.warning("Bu konfigürasyon klasik güçlü düzenlilik (SRG) koşullarını tam olarak sağlamamaktadır.")

st.markdown("---")

# Sekmeli Detay Görünümü
tab1, tab2, tab3 = st.tabs(["📊 Graf Görselleştirme", "🔢 Latin Kareler ve OA Matrisi", "📈 Spektral Özdeğer Analizi"])

with tab1:
    st.subheader("NetworkX Topoloji Görselleştirmesi")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if layout_style == "Spring Layout":
        pos = nx.spring_layout(G, seed=42)
    elif layout_style == "Circular Layout":
        pos = nx.circular_layout(G)
    else:
        pos = nx.shell_layout(G)
        
    nx.draw(
        G, pos, ax=ax,
        node_color='skyblue', node_size=500,
        edge_color='gray', with_labels=True,
        font_weight='bold', font_size=10
    )
    st.pyplot(fig)

with tab2:
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.write("#### Latin Kare 1 ($L_1$)")
        st.dataframe(L1, use_container_width=True)
    with col_l2:
        st.write("#### Latin Kare 2 ($L_2$)")
        st.dataframe(L2, use_container_width=True)
        
    st.write("#### Ortogonal Dizi $OA(4, n)$ Örneklem Sütunları")
    st.dataframe(oa[:min(10, len(oa))], use_container_width=True)

with tab3:
    st.subheader("Adjacency Matris Özdeğer Spektrumu")
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    eigenvalues = np.linalg.eigvalsh(A)
    
    st.write("Hesaplanan Özdeğerler (Ascending):")
    st.code(np.round(eigenvalues, 4))
    
    fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
    ax_spec.hist(eigenvalues, bins=10, color='teal', edgecolor='black')
    ax_spec.set_title("Özdeğer Dağılım Histogramı")
    ax_spec.set_xlabel("Özdeğer Değeri")
    ax_spec.set_ylabel("Frekans")
    st.pyplot(fig_spec)
