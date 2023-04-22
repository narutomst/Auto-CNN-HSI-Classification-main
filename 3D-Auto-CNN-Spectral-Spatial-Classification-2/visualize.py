import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', ),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', ),
        engine='dot')
    # node_attr中的fontname='times'，edge_attr中的fontname='times'报错；
    # 最终解决方案：删除 fontname 关键字参数，这样 node_attr 和 edge_attr 将会使用font的默认值
    # 默认字体格式：fontname="Times-Roman", fontsize=14, fontcolor=black

    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # 打印生成符合Dot语法规范的源代码，可以将源代码复制粘贴到Graphviz Visual Editor（http://magjac.com/graphviz-visual-editor）
    # 的代码区（位于页面左半边），绘图区（页面右半边）就会自动绘制图。最后将图截屏保存
    print(g.source)

    g.render(filename, view=True, format='png')


if __name__ == '__main__':
    genotype_name = 'HSI'
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
