from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise"
DEFAULT_OUTPUT = DEFAULT_SOURCE_DIR / "manifest.json"
DEFAULT_USER_AGENT = "doc-helper-zh-enterprise-benchmark/1.0"
HTML_BLOCK_TAGS = {
    "article",
    "body",
    "br",
    "dd",
    "div",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
HTML_SKIP_TAGS = {"button", "footer", "form", "header", "nav", "noscript", "script", "select", "style", "svg"}
UI_NOISE_EXACT = {
    "登录",
    "注册",
    "首页",
    "搜索",
    "分享",
    "收藏",
    "打印",
    "关闭",
    "微博",
    "微信",
    "客户端",
    "手机版",
    "无障碍",
    "简",
    "繁",
    "en",
    "个人中心",
    "退出",
    "邮箱",
    "默认",
    "大",
    "超大",
    "留言",
    "政策",
    "国务院政策文件库",
    "关于本网",
    "联系我们",
    "网站纠错",
    "电脑版",
    "小程序",
    "国务院客户端",
    "国务院客户端小程序",
    "全国人大",
    "全国政协",
    "国家监察委员会",
    "最高人民法院",
    "最高人民检察院",
    "解读",
    "×",
    "x",
    "学习强国",
    "返回顶部",
    "PC版",
}
UI_NOISE_PATTERNS = (
    "版权所有",
    "ICP备案",
    "网站标识码",
    "责任编辑",
    "字号：",
    "当前位置",
    "扫一扫",
    "打开微信",
    "二维码",
    "微博微信",
    "分享至",
    "中国政府网",
    "国务院文件",
    "政策文件库",
    "索 引 号",
    "主题分类",
    "发文机关",
    "成文日期",
    "标 题",
    "发布日期",
    "京公网安备",
    "Produced By CMS",
)
FOOTER_START_KEYS = {
    "链接",
    "相关链接",
    "友情链接",
    "底部",
    "网站地图",
    "解读",
    "中央网络安全和信息化委员会办公室",
    "学习强国",
    "返回顶部",
    "pc版",
    "打开微信",
}


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    url: str
    description: str
    expected_facts: list[str]
    questions: list[str]
    acl: list[dict[str, Any]]
    metadata: dict[str, Any]
    evidence_aliases: list[list[str]] | None = None


@dataclass(frozen=True)
class CompositeCase:
    case_name: str
    acting_user_email: str
    question: str
    source_fact_indexes: list[tuple[str, int]]
    scoring_notes: str
    metadata: dict[str, Any]


SOURCES = [
    SourceDocument(
        id="green_factory_policy",
        title="zh_enterprise:policy:绿色工厂梯度培育及管理暂行办法",
        url="https://www.gov.cn/zhengce/zhengceku/202401/P020240130421561581025.pdf",
        description="工业和信息化部发布的绿色工厂梯度培育及管理暂行办法 PDF。",
        expected_facts=[
            "绿色工厂梯度培育及管理遵循企业主体、政府引导、标准引领和全面覆盖的原则",
            "绿色工厂梯度培育从纵向和横向两个维度建立培育机制：纵向形成国家、省、市三级联动的绿色工厂培育机制，横向形成绿色工业园区、绿色供应链管理企业带动园区内、供应链上下游企业创建绿色工厂的培育机制",
            "绿色工厂是指实现用地集约化、原料无害化、生产洁净化、废物资源化、能源低碳化的企业，是绿色制造核心实施单元",
            "企业、园区可采取自评价或委托具备评价能力的第三方机构开展评价的方式，编写评价报告后通过管理平台提交；第三方机构对所出具评价报告的真实性和准确性负责",
            "省级工业和信息化主管部门应在每年7月31日前将具有代表性和引领性的省层面绿色工厂通过管理平台推荐至工业和信息化部，工业和信息化部评审后公示15日，无异议的纳入国家层面绿色工厂名单",
            "国家、省、市层面绿色制造名单应在每年4月15日前通过管理平台填报动态管理表，上报年度绿色制造关键指标情况",
        ],
        questions=[
            "绿色工厂梯度培育及管理遵循哪些原则？",
            "绿色工厂梯度培育从哪两个维度建立培育机制？",
            "办法中对绿色工厂本身是怎么定义的？",
            "绿色工厂申报评价可以采用哪些方式，第三方评价机构要承担什么责任？",
            "国家层面绿色工厂推荐和公示的关键时间要求是什么？",
            "绿色制造名单动态管理表应在什么时候填报，主要上报什么？",
        ],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "policy",
            "source_org": "工业和信息化部",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_qa",
        },
        evidence_aliases=[
            ["企业主体、政府引导、标准引领和全面覆盖"],
            ["绿色工厂梯度培育是指从以下两个维度建立培育机制", "横向形成绿色工业园区、绿色供应链管理企业"],
            ["实现用地集约化、原料无害化、生产洁净化、废物资源化、能源低碳化", "绿色制造核心实施单元"],
            ["可采取自评价或委托具备评价能力的第三方机构开展评价", "对所出具评价报告的真实性和准确性负责"],
            ["于每年 7 月 31 日前", "公示时间为 15 日", "纳入国家层面绿色工厂名单"],
            ["每年 4 月 15 日前通过管理平台填报动态管理表", "上报年度绿色制造关键指标情况"],
        ],
    ),
    SourceDocument(
        id="state_owned_manager_discipline",
        title="zh_enterprise:policy:国有企业管理人员处分条例",
        url="https://www.gov.cn/zhengce/zhengceku/202405/content_6954056.htm",
        description="国务院公布的国有企业管理人员处分条例 HTML。",
        expected_facts=[
            "国有企业管理人员处分工作坚持中国共产党的领导",
            "处分的种类包括警告、记过、记大过、降级、撤职和开除",
            "对涉嫌违法的国有企业管理人员进行调查、处理，应当由2名以上工作人员进行，并经过初步核实、立案告知、调查取证、听取陈述申辩、集体讨论决定、书面通知和归档等程序",
            "决定给予处分的，应当制作处分决定书，载明被处分人的姓名、工作单位和职务，违法事实和证据，处分的种类和依据，复核申诉途径和期限，以及作出决定的机关、单位名称和日期",
            "国有企业管理人员已经被立案调查且不宜继续履行职责的，可以暂停其履行职务；立案调查期间未经同意不得出境、辞去公职，也不得办理交流、晋升、奖励或者退休手续",
            "被处分人对处分决定不服的，可以自收到处分决定书之日起1个月内申请复核；对复核决定仍不服的，可以自收到复核决定之日起1个月内申诉，申诉机关一般应自受理之日起2个月以内作出处理决定",
            "国有企业管理人员受到开除以外的处分，在受处分期间有悔改表现且没有再出现应当给予处分的违法情形的，处分期满后自动解除处分",
        ],
        questions=[
            "国有企业管理人员处分工作应当坚持什么原则？",
            "国有企业管理人员处分的种类包括哪些？",
            "对涉嫌违法的国有企业管理人员调查处理时，程序上有哪些关键步骤？",
            "国有企业管理人员处分决定书应当载明哪些事项？",
            "国有企业管理人员被立案调查期间，任免机关可以采取哪些履职和人事限制？",
            "被处分人不服处分决定时，复核和申诉期限分别怎么规定？",
            "国有企业管理人员受到开除以外处分后，什么条件下处分期满自动解除？",
        ],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "policy",
            "source_org": "国务院",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_permission",
        },
        evidence_aliases=[
            ["坚持中国共产党的领导，坚持党管干部原则"],
            ["处分的种类为"],
            ["应当由2名以上工作人员进行", "经初步核实", "听取其陈述和申辩", "将处分有关决定及执行材料归入"],
            ["决定给予处分的，应当制作处分决定书", "违法事实和证据", "申请复核、申诉的途径和期限"],
            ["不宜继续履行职责的", "可以决定暂停其履行职务", "不得出境、辞去公职", "不得对其交流、晋升、奖励或者办理退休手续"],
            ["自收到处分决定书之日起1个月内", "自收到复核决定之日起1个月内", "自受理之日起2个月以内"],
            ["受到开除以外的处分", "处分期满后自动解除处分"],
        ],
    ),
    SourceDocument(
        id="data_cross_border_rules",
        title="zh_enterprise:policy:促进和规范数据跨境流动规定",
        url="https://www.cac.gov.cn/2024-03/22/c_1712776611775634.htm",
        description="国家互联网信息办公室发布的促进和规范数据跨境流动规定 HTML。",
        expected_facts=[
            "数据处理者向境外提供重要数据应当通过所在地省级网信部门向国家网信部门申报数据出境安全评估",
            "关键信息基础设施运营者向境外提供个人信息或者重要数据应当依法申报数据出境安全评估",
            "国际贸易、跨境运输、学术合作、跨国生产制造和市场营销等活动中收集和产生的数据向境外提供，不包含个人信息或者重要数据的，免予申报数据出境安全评估、订立个人信息出境标准合同、通过个人信息保护认证",
            "按照依法制定的劳动规章制度和依法签订的集体合同实施跨境人力资源管理，确需向境外提供员工个人信息的，免予申报数据出境安全评估、订立个人信息出境标准合同、通过个人信息保护认证",
            "关键信息基础设施运营者以外的数据处理者自当年1月1日起累计向境外提供10万人以上、不满100万人个人信息或者不满1万人敏感个人信息的，应当依法订立个人信息出境标准合同或者通过个人信息保护认证",
            "通过数据出境安全评估的结果有效期为3年；有效期届满前60个工作日内可以申请延长，经批准可以延长3年",
        ],
        questions=[
            "数据处理者向境外提供重要数据时，应当如何申报安全评估？",
            "关键信息基础设施运营者向境外提供个人信息或者重要数据有什么要求？",
            "国际贸易、跨境运输等业务场景向境外提供数据时，在什么情况下可以免予数据出境合规手续？",
            "企业做跨境人力资源管理时，员工个人信息出境是否可以免予申报或认证？条件是什么？",
            "非关键信息基础设施运营者达到什么个人信息出境规模时，应订立标准合同或通过个人信息保护认证？",
            "数据出境安全评估结果有效期和延长申请期限是怎么规定的？",
        ],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国家互联网信息办公室",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_permission",
        },
        evidence_aliases=[
            ["通过所在地省级网信部门向国家网信部门申报数据出境安全评估"],
            ["关键信息基础设施运营者向境外提供个人信息或者重要数据"],
            ["国际贸易、跨境运输、学术合作、跨国生产制造和市场营销", "不包含个人信息或者重要数据", "免予申报数据出境安全评估"],
            ["跨境人力资源管理", "确需向境外提供员工个人信息", "免予申报数据出境安全评估"],
            ["10万人以上、不满100万人个人信息", "不满1万人敏感个人信息", "订立个人信息出境标准合同或者通过个人信息保护认证"],
            ["结果有效期为3年", "有效期届满前60个工作日内", "可以延长评估结果有效期3年"],
        ],
    ),
    SourceDocument(
        id="ai_service_rules",
        title="zh_enterprise:policy:生成式人工智能服务管理暂行办法",
        url="https://www.gov.cn/zhengce/zhengceku/202307/content_6891752.htm",
        description="生成式人工智能服务管理暂行办法 HTML。",
        expected_facts=[
            "提供和使用生成式人工智能服务应当遵守法律、行政法规，尊重社会公德和伦理道德",
            "提供者应当依法承担网络信息内容生产者责任，履行网络信息安全义务",
            "提供者开展预训练、优化训练等训练数据处理活动，应当使用具有合法来源的数据和基础模型，不得侵害知识产权，涉及个人信息的应取得个人同意或者符合法定情形，并采取有效措施提高训练数据质量",
            "提供者应当与注册使用者签订服务协议，明确双方权利义务，并明确公开其服务的适用人群、场合和用途",
            "提供者对使用者的输入信息和使用记录应当依法履行保护义务，不得收集非必要个人信息，不得非法留存能够识别使用者身份的输入信息和使用记录，也不得非法向他人提供",
            "提供者发现违法内容的，应当及时采取停止生成、停止传输、消除等处置措施，采取模型优化训练等措施整改，并向有关主管部门报告",
            "提供具有舆论属性或者社会动员能力的生成式人工智能服务，应当按照国家有关规定开展安全评估，并履行算法备案和变更、注销备案手续",
        ],
        questions=[
            "提供和使用生成式人工智能服务应当遵守哪些基本要求？",
            "生成式人工智能服务提供者需要承担哪些责任？",
            "生成式人工智能服务提供者在训练数据处理上需要遵守哪些要求？",
            "生成式人工智能服务提供者与注册使用者之间的服务协议和适用范围公开有什么要求？",
            "生成式人工智能服务提供者应如何保护用户输入信息和使用记录？",
            "生成式人工智能服务提供者发现违法内容时应当如何处置？",
            "具有舆论属性或社会动员能力的生成式人工智能服务需要履行哪些安全评估和备案要求？",
        ],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "ai_governance",
            "source_org": "国家互联网信息办公室等",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_qa",
        },
        evidence_aliases=[
            ["遵守法律、行政法规，尊重社会公德和伦理道德"],
            ["依法承担网络信息内容生产者责任，履行网络信息安全义务"],
            ["使用具有合法来源的数据和基础模型", "不得侵害他人依法享有的知识产权", "提高训练数据质量"],
            ["签订服务协议，明确双方权利义务", "明确并公开其服务的适用人群、场合、用途"],
            ["输入信息和使用记录应当依法履行保护义务", "不得收集非必要个人信息", "不得非法留存能够识别使用者身份的输入信息和使用记录"],
            ["发现违法内容的", "停止生成、停止传输、消除", "采取模型优化训练等措施进行整改"],
            ["具有舆论属性或者社会动员能力", "开展安全评估", "履行算法备案和变更、注销备案手续"],
        ],
    ),
    SourceDocument(
        id="enterprise_data_accounting",
        title="zh_enterprise:finance:企业数据资源相关会计处理暂行规定",
        url="https://www.mof.gov.cn/gkml/caizhengwengao/wg2023/wg202308/202312/P020231227577411941131.pdf",
        description="财政部企业数据资源相关会计处理暂行规定 PDF。",
        expected_facts=[
            "企业应当按照企业会计准则相关规定，根据数据资源的持有目的、形成方式、业务模式以及预期消耗方式等对数据资源相关交易和事项进行会计处理",
            "企业使用的数据资源符合无形资产准则规定的定义和确认条件的，应当确认为无形资产",
            "企业通过外购方式取得确认为无形资产的数据资源，其成本包括购买价款、相关税费、直接归属于达到预定用途所发生的数据脱敏、清洗、标注、整合、分析、可视化等加工支出，以及数据权属鉴证、质量评估、登记结算、安全管理等费用",
            "企业内部数据资源研究开发项目的支出应当区分研究阶段和开发阶段；研究阶段支出应当于发生时计入当期损益，开发阶段支出满足条件的才能确认为无形资产",
            "企业日常活动中持有、最终目的用于出售的数据资源，符合存货准则规定的定义和确认条件的，应当确认为存货",
            "企业编制资产负债表时，应当根据重要性原则在存货、无形资产和开发支出项目下增设其中：数据资源项目，分别反映相关数据资源的期末账面价值或资本化支出金额",
            "企业可以自愿披露数据资源的应用场景或业务模式、原始数据类型规模来源权属质量、加工维护和安全保护情况、应用情况、重大交易影响及风险、权利失效和权利限制等信息",
        ],
        questions=[
            "企业对数据资源相关交易和事项进行会计处理时，应考虑哪些因素？",
            "企业使用的数据资源在什么情况下应当确认为无形资产？",
            "企业外购并确认为无形资产的数据资源，成本通常包括哪些支出？",
            "企业内部数据资源研发项目的研究阶段和开发阶段支出应如何处理？",
            "企业持有的数据资源在什么情况下应当确认为存货？",
            "企业在资产负债表中如何列示数据资源相关项目？",
            "企业可以自愿披露哪些数据资源相关信息？",
        ],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "finance",
            "source_org": "财政部",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_qa",
        },
        evidence_aliases=[
            ["持有目的、形成方式、业务模式", "预期消耗方式"],
            ["应当确认为无形资产"],
            ["购买价款、相关税费", "数据脱敏、清洗、标注、整合", "数据权属鉴证、质量评估、登记结算、安全管理"],
            ["区分研究阶段支出与开发阶段支出", "研究阶段的支出，应当于发生时计入当期损益", "开发阶段的支出，满足"],
            ["日常活动中持有、最终目的用于出售的数据资源", "应当确认为存货"],
            ["在“存货”项目下增设“其中：数据资源”项目", "在“无形资产”项目下增设“其中：数据资源”", "在“开发支出”项目下增设“其中：数据资源”"],
            ["可以根据实际情况，自愿披露", "应用场景或业务模式", "加工维护和安全保护情况", "权利限制"],
        ],
    ),
    SourceDocument(
        id="network_data_security_rules",
        title="zh_enterprise:data_security:网络数据安全管理条例",
        url="https://www.gov.cn/zhengce/zhengceku/202409/content_6977767.htm",
        description="国务院公布的网络数据安全管理条例 HTML。",
        expected_facts=[
            "国家根据网络数据在经济社会发展中的重要程度，以及遭到篡改、破坏、泄露或者非法获取、非法利用后的危害程度，对网络数据实行分类分级保护",
            "网络数据处理者应当在网络安全等级保护的基础上加强网络数据安全防护，建立健全网络数据安全管理制度并采取必要技术措施",
            "网络数据处理者发现网络产品、服务存在安全缺陷、漏洞等风险时，应立即采取补救措施，按规定及时告知用户并向有关主管部门报告；涉及危害国家安全、公共利益的，还应在24小时内报告",
            "网络数据处理者应当建立健全网络数据安全事件应急预案，发生网络数据安全事件时立即启动预案，采取措施防止危害扩大、消除安全隐患，并按规定报告",
            "网络数据处理者向其他网络数据处理者提供、委托处理个人信息和重要数据的，应当通过合同等约定处理目的、方式、范围以及安全保护义务，并对处理情况记录至少保存3年",
            "网络数据处理者处理个人信息前通过个人信息处理规则告知的，规则应集中公开展示、易于访问、置于醒目位置，并明确处理者信息、处理目的方式种类、保存期限和个人行使权利的方法途径等内容",
            "重要数据的处理者应当每年度对网络数据处理活动开展风险评估，并向省级以上有关主管部门报送风险评估报告",
            "网络数据处理者在中华人民共和国境内运营中收集和产生的重要数据确需向境外提供的，应当通过国家网信部门组织的数据出境安全评估",
        ],
        questions=[
            "国家根据什么对网络数据实行分类分级保护？",
            "网络数据处理者应当在什么基础上加强网络数据安全防护？",
            "网络产品或服务存在安全缺陷、漏洞时，网络数据处理者应如何处理和报告？",
            "发生网络数据安全事件时，网络数据处理者应当如何启动预案和处置？",
            "企业向其他处理者提供或委托处理个人信息和重要数据时，合同约定和记录保存有什么要求？",
            "个人信息处理规则应如何展示，并至少包含哪些内容？",
            "重要数据处理者年度风险评估报告的报送要求是什么？",
            "境内运营中收集和产生的重要数据确需出境时，应履行什么安全评估要求？",
        ],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国务院",
            "language": "zh",
            "benchmark_profile": "zh_enterprise_permission",
        },
        evidence_aliases=[
            ["对网络数据实行分类分级保护"],
            ["在网络安全等级保护的基础上", "建立健全网络数据安全管理制度"],
            ["安全缺陷、漏洞等风险", "立即采取补救措施", "还应当在24小时内向有关主管部门报告"],
            ["建立健全网络数据安全事件应急预案", "发生网络数据安全事件时，应当立即启动预案"],
            ["通过合同等与网络数据接收方约定处理目的、方式、范围", "记录应当至少保存3年"],
            ["集中公开展示、易于访问并置于醒目位置", "个人信息保存期限", "个人查阅、复制、转移、更正"],
            ["每年度对其网络数据处理活动开展风险评估", "报送风险评估报告"],
            ["在中华人民共和国境内运营中收集和产生的重要数据", "应当通过国家网信部门组织的数据出境安全评估"],
        ],
    ),
]


COMPOSITE_CASES = [
    CompositeCase(
        case_name="green_factory_policy:multi:recommend_and_dynamic_deadlines",
        acting_user_email="viewer@local.test",
        question="国家层面绿色工厂推荐、公示和动态管理表填报分别有哪些时间要求？",
        source_fact_indexes=[("green_factory_policy", 4), ("green_factory_policy", 5)],
        scoring_notes="同一真实 PDF 长文档多证据问题：同时要求推荐/公示时间和动态管理表填报时间。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "green_factory_policy"},
    ),
    CompositeCase(
        case_name="state_owned_manager_discipline:multi:investigation_and_decision_notice",
        acting_user_email="manager@local.test",
        question="对涉嫌违法的国有企业管理人员，从调查处理程序到处分决定书内容，企业需要同时注意哪些要求？",
        source_fact_indexes=[("state_owned_manager_discipline", 2), ("state_owned_manager_discipline", 3)],
        scoring_notes="同一 manager 权限长文档多证据问题：同时覆盖调查程序和处分决定书内容。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "state_owned_manager_discipline"},
    ),
    CompositeCase(
        case_name="data_cross_border_rules:multi:hr_and_standard_contract_threshold",
        acting_user_email="manager@local.test",
        question="企业做跨境人力资源管理和达到较大个人信息出境规模时，分别适用哪些合规要求？",
        source_fact_indexes=[("data_cross_border_rules", 3), ("data_cross_border_rules", 4)],
        scoring_notes="同一 manager 权限长文档多证据问题：同时覆盖员工个人信息出境豁免和标准合同/认证阈值。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "data_cross_border_rules"},
    ),
    CompositeCase(
        case_name="ai_service_rules:multi:user_records_and_illegal_content",
        acting_user_email="viewer@local.test",
        question="生成式人工智能服务提供者对用户输入、使用记录和违法内容分别应当怎么处理？",
        source_fact_indexes=[("ai_service_rules", 4), ("ai_service_rules", 5)],
        scoring_notes="同一真实 HTML 清洗文档多证据问题：同时覆盖用户记录保护和违法内容处置。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "ai_service_rules"},
    ),
    CompositeCase(
        case_name="enterprise_data_accounting:multi:intangible_cost_and_rd_stage",
        acting_user_email="viewer@local.test",
        question="企业数据资源确认为无形资产时，外购成本和内部研发阶段支出分别应如何处理？",
        source_fact_indexes=[("enterprise_data_accounting", 2), ("enterprise_data_accounting", 3)],
        scoring_notes="同一真实 PDF 长文档多证据问题：同时覆盖外购无形资产成本和内部研发阶段处理。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "enterprise_data_accounting"},
    ),
    CompositeCase(
        case_name="network_data_security_rules:multi:vulnerability_and_incident_response",
        acting_user_email="manager@local.test",
        question="网络产品服务存在漏洞风险和发生网络数据安全事件时，处理者分别应采取哪些处置与报告措施？",
        source_fact_indexes=[("network_data_security_rules", 2), ("network_data_security_rules", 3)],
        scoring_notes="同一 manager 权限长文档多证据问题：同时覆盖漏洞风险补救报告和安全事件应急处置。",
        metadata={"case_type": "multi_evidence_same_document", "source_id": "network_data_security_rules"},
    ),
    CompositeCase(
        case_name="cross_data_security:multi:important_data_export_assessment",
        acting_user_email="manager@local.test",
        question="重要数据出境在《促进和规范数据跨境流动规定》和《网络数据安全管理条例》中分别有哪些申报或评估要求？",
        source_fact_indexes=[("data_cross_border_rules", 0), ("network_data_security_rules", 7)],
        scoring_notes="跨两个真实 manager 权限文档的问题：同时要求两个制度中的重要数据出境要求。",
        metadata={"case_type": "multi_evidence_cross_document", "source_ids": ["data_cross_border_rules", "network_data_security_rules"]},
    ),
    CompositeCase(
        case_name="cross_data_security:multi:ai_user_records_and_personal_info_rules",
        acting_user_email="manager@local.test",
        question="生成式人工智能服务和网络数据处理活动中，涉及用户记录或个人信息处理规则时分别有哪些保护或公开要求？",
        source_fact_indexes=[("ai_service_rules", 4), ("network_data_security_rules", 5)],
        scoring_notes="跨公开文档和 manager 权限文档的问题：同时测试多文档证据召回和权限内组合回答。",
        metadata={"case_type": "multi_evidence_cross_document", "source_ids": ["ai_service_rules", "network_data_security_rules"]},
    ),
]


LOW_OVERLAP_SCENARIO_CASES = [
    CompositeCase(
        case_name="green_factory_policy:low_overlap:annual_roster_maintenance",
        acting_user_email="viewer@local.test",
        question="工厂已经进入绿色制造名单后，4月中旬前还要在平台补报哪些年度运行数据？",
        source_fact_indexes=[("green_factory_policy", 5)],
        scoring_notes="低词面重合企业场景题：年度维护问法，证据仍来自同一真实 PDF 长文档。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "green_factory_policy",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="green_factory_policy:low_overlap:green_unit_assessment",
        acting_user_email="viewer@local.test",
        question="集团评估一个生产基地是否算绿色制造核心单元时，应从土地、原料、生产、废弃物和能源五方面看哪些状态？",
        source_fact_indexes=[("green_factory_policy", 2)],
        scoring_notes="低词面重合企业场景题：把定义改写为集团评估口径，测试 PDF 定义段召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "green_factory_policy",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="state_owned_manager_discipline:low_overlap:investigation_job_freeze",
        acting_user_email="manager@local.test",
        question="管理人员被立案调查后，公司认为其暂时不适合继续任职，岗位、出境、离职和晋升退休手续能怎么管控？",
        source_fact_indexes=[("state_owned_manager_discipline", 4)],
        scoring_notes="低词面重合企业场景题：人事管控问法，测试 manager 权限 HTML 长文档召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "state_owned_manager_discipline",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="state_owned_manager_discipline:low_overlap:appeal_timeline",
        acting_user_email="manager@local.test",
        question="处分结果出来后本人不认可，内部复核、继续申诉以及受理机关处理分别卡在哪些期限？",
        source_fact_indexes=[("state_owned_manager_discipline", 5)],
        scoring_notes="低词面重合企业场景题：员工申诉流程问法，测试期限类证据召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "state_owned_manager_discipline",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="data_cross_border_rules:low_overlap:overseas_hr_system",
        acting_user_email="manager@local.test",
        question="把员工资料同步到海外 HR 系统时，满足什么前提可以不用安全评估、标准合同或个人信息保护认证？",
        source_fact_indexes=[("data_cross_border_rules", 3)],
        scoring_notes="低词面重合企业场景题：海外 HR 系统问法，测试跨境人力资源管理豁免证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "data_cross_border_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="data_cross_border_rules:low_overlap:business_data_export_exemption",
        acting_user_email="manager@local.test",
        question="海外贸易、运输或营销项目只传业务数据，不含个人信息和重要数据时，三类出境手续是否还能豁免？",
        source_fact_indexes=[("data_cross_border_rules", 2)],
        scoring_notes="低词面重合企业场景题：业务项目豁免问法，测试多场景枚举证据召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "data_cross_border_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="data_cross_border_rules:low_overlap:customer_volume_contract_cert",
        acting_user_email="manager@local.test",
        question="非关基单位今年向境外共享的客户个人信息已经超过十万但还不到百万，通常要走标准合同还是认证？",
        source_fact_indexes=[("data_cross_border_rules", 4)],
        scoring_notes="低词面重合企业场景题：客户规模阈值问法，测试数量条件和合规路径召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "data_cross_border_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="ai_service_rules:low_overlap:input_log_retention",
        acting_user_email="viewer@local.test",
        question="AI 助手上线后，平台保存用户提示词和操作痕迹时有哪些收集、留存、对外提供方面的红线？",
        source_fact_indexes=[("ai_service_rules", 4)],
        scoring_notes="低词面重合企业场景题：提示词和操作痕迹问法，测试用户输入/使用记录保护证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "ai_service_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="ai_service_rules:low_overlap:bad_output_response",
        acting_user_email="viewer@local.test",
        question="模型输出了违法内容，服务方除了停止继续展示，还要做哪些整改和上报动作？",
        source_fact_indexes=[("ai_service_rules", 5)],
        scoring_notes="低词面重合企业场景题：坏输出处置问法，测试违法内容处理证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "ai_service_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="enterprise_data_accounting:low_overlap:purchased_dataset_capitalization",
        acting_user_email="viewer@local.test",
        question="采购来的数据集准备作为无形资产入账，除购买价外，清洗标注、权属质量和安全管理等费用怎么计入成本？",
        source_fact_indexes=[("enterprise_data_accounting", 2)],
        scoring_notes="低词面重合企业场景题：采购数据资产入账问法，测试 PDF 成本构成证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "enterprise_data_accounting",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="enterprise_data_accounting:low_overlap:self_developed_dataset_stage",
        acting_user_email="viewer@local.test",
        question="自研数据产品从探索到开发，哪些阶段费用直接进当期损益，哪些满足条件后才能资本化？",
        source_fact_indexes=[("enterprise_data_accounting", 3)],
        scoring_notes="低词面重合企业场景题：自研数据产品资本化问法，测试研发阶段处理证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "enterprise_data_accounting",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="network_data_security_rules:low_overlap:vulnerability_public_interest",
        acting_user_email="manager@local.test",
        question="安全团队发现产品漏洞且可能影响公共利益时，除了修补和通知用户，最迟多久向主管部门报告？",
        source_fact_indexes=[("network_data_security_rules", 2)],
        scoring_notes="低词面重合企业场景题：漏洞处置问法，测试 24 小时报告证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "network_data_security_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="network_data_security_rules:low_overlap:data_breach_playbook",
        acting_user_email="manager@local.test",
        question="发生疑似数据泄露事故后，处理者启动预案时要优先做哪些止损、隐患消除和报告动作？",
        source_fact_indexes=[("network_data_security_rules", 3)],
        scoring_notes="低词面重合企业场景题：数据泄露事故问法，测试安全事件应急预案证据。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_id": "network_data_security_rules",
            "query_style": "scenario_paraphrase",
        },
    ),
    CompositeCase(
        case_name="cross_data_security:low_overlap:ai_logs_and_privacy_notice",
        acting_user_email="manager@local.test",
        question="公司同时运营 AI 助手和数据平台，关于用户操作痕迹保存以及个人信息规则公开，两套制度分别要求什么？",
        source_fact_indexes=[("ai_service_rules", 4), ("network_data_security_rules", 5)],
        scoring_notes="低词面重合跨文档场景题：同时测试公开 HTML 文档和 manager 权限 HTML 文档的多证据召回。",
        metadata={
            "case_type": "low_overlap_enterprise_scenario",
            "source_ids": ["ai_service_rules", "network_data_security_rules"],
            "query_style": "scenario_paraphrase",
        },
    ),
]


DISTRACTOR_SOURCES = [
    SourceDocument(
        id="industrial_data_security_rules",
        title="zh_enterprise:data_security:工业和信息化领域数据安全管理办法试行",
        url="https://www.gov.cn/zhengce/zhengceku/2022-12/13/content_5731663.htm",
        description="工业和信息化部发布的工业和信息化领域数据安全管理办法（试行）HTML，用作同域数据安全干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "工业和信息化部",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="data_export_security_assessment",
        title="zh_enterprise:data_security:数据出境安全评估办法",
        url="https://www.cac.gov.cn/2022-07/07/c_1658811536396503.htm",
        description="国家互联网信息办公室发布的数据出境安全评估办法 HTML，用作跨境数据合规干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国家互联网信息办公室",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="personal_info_export_standard_contract",
        title="zh_enterprise:data_security:个人信息出境标准合同办法",
        url="https://www.cac.gov.cn/2023-02/24/c_1678884830036813.htm",
        description="国家互联网信息办公室发布的个人信息出境标准合同办法 HTML，用作跨境个人信息合规干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国家互联网信息办公室",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="network_security_review_rules",
        title="zh_enterprise:data_security:网络安全审查办法",
        url="https://www.cac.gov.cn/2022-01/04/c_1642894602182845.htm",
        description="国家互联网信息办公室等部门发布的网络安全审查办法 HTML，用作网络安全合规干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国家互联网信息办公室等",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="critical_infrastructure_security_rules",
        title="zh_enterprise:data_security:关键信息基础设施安全保护条例",
        url="https://www.gov.cn/zhengce/content/2021-08/17/content_5631671.htm",
        description="国务院公布的关键信息基础设施安全保护条例 HTML，用作关基保护干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "国务院",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="industrial_data_security_capability_plan",
        title="zh_enterprise:data_security:工业领域数据安全能力提升实施方案2024至2026年",
        url="https://www.gov.cn/zhengce/zhengceku/202402/P020240227202289762834.pdf",
        description="工业和信息化部工业领域数据安全能力提升实施方案 PDF，用作工业企业数据安全干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "data_security",
            "source_org": "工业和信息化部",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="manufacturing_digital_transformation_guide",
        title="zh_enterprise:digital_transformation:制造业企业数字化转型实施指南",
        url="https://www.gov.cn/zhengce/zhengceku/202412/P020241226453066361975.pdf",
        description="工业和信息化部等部门制造业企业数字化转型实施指南 PDF，用作制造业企业数字化转型干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "digital_transformation",
            "source_org": "工业和信息化部等",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="sme_digital_assessment_2024",
        title="zh_enterprise:digital_transformation:中小企业数字化水平评测指标2024年版",
        url="https://www.gov.cn/zhengce/zhengceku/202409/P020240910515398355720.pdf",
        description="工业和信息化部中小企业数字化水平评测指标 2024 年版 PDF，用作企业数字化评估干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "digital_transformation",
            "source_org": "工业和信息化部",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="algorithm_recommendation_rules",
        title="zh_enterprise:ai_governance:互联网信息服务算法推荐管理规定",
        url="https://www.miit.gov.cn/jgsj/waj/wjfb/art/2022/art_a6ae77ea1f5e401eb8cc6819b869fdfa.html",
        description="国家网信办等部门发布的互联网信息服务算法推荐管理规定 HTML，用作 AI/算法治理干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "ai_governance",
            "source_org": "国家互联网信息办公室等",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="central_enterprise_compliance_rules",
        title="zh_enterprise:governance:中央企业合规管理办法",
        url="http://www.sasac.gov.cn/n2588035/n2588320/n2588335/c26018430/content.html",
        description="国务院国资委发布的中央企业合规管理办法 HTML，用作企业合规治理干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "governance",
            "source_org": "国务院国资委",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="central_enterprise_overseas_investment_rules",
        title="zh_enterprise:governance:中央企业境外投资监督管理办法",
        url="http://www.sasac.gov.cn/n2588035/n2588320/n2588335/c20164457/content.html",
        description="国务院国资委发布的中央企业境外投资监督管理办法 HTML，用作境外投资与风险管理干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "governance",
            "source_org": "国务院国资委",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="central_enterprise_internal_control_guidance",
        title="zh_enterprise:governance:加强中央企业内部控制体系建设与监督工作的实施意见",
        url="http://www.sasac.gov.cn/n2588035/n2588320/n2588335/c12670064/content.html",
        description="国务院国资委发布的中央企业内部控制体系建设与监督实施意见 HTML，用作内控与风险监督干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "role", "role_name": "manager"}],
        metadata={
            "domain": "governance",
            "source_org": "国务院国资委",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="small_enterprise_internal_control_rules",
        title="zh_enterprise:finance:小企业内部控制规范试行",
        url="https://kjs.mof.gov.cn/zhengcefabu/201707/t20170707_2640522.htm",
        description="财政部发布的小企业内部控制规范（试行）HTML，用作财务内控干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "finance",
            "source_org": "财政部",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
    SourceDocument(
        id="industrial_carbon_peak_plan",
        title="zh_enterprise:green_manufacturing:工业领域碳达峰实施方案",
        url="https://www.gov.cn/zhengce/zhengceku/2022-08/01/5703910/files/f7edf770241a404c9bc608c051f13b45.pdf",
        description="工业和信息化部等部门工业领域碳达峰实施方案 PDF，用作绿色制造与工业低碳干扰文档。",
        expected_facts=[],
        questions=[],
        acl=[{"principal_type": "public"}],
        metadata={
            "domain": "green_manufacturing",
            "source_org": "工业和信息化部等",
            "language": "zh",
            "benchmark_role": "distractor",
        },
    ),
]


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir).resolve()
    manifest = build_manifest(
        source_dir,
        download=not args.skip_download,
        dataset_name=args.dataset_name,
        user_agent=args.user_agent,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"documents={len(manifest['documents'])} cases={len(manifest['cases'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download real Chinese enterprise/policy documents and build a benchmark manifest."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset-name", default="zh_enterprise_real")
    parser.add_argument("--skip-download", action="store_true", help="Only build manifest from existing files.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def build_manifest(
    source_dir: Path,
    *,
    download: bool,
    dataset_name: str,
    user_agent: str,
) -> dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    cases = []
    title_by_id: dict[str, str] = {}
    all_sources = [*SOURCES, *DISTRACTOR_SOURCES]
    source_by_id = {source.id: source for source in all_sources}

    for source in all_sources:
        local_path = prepare_source_file(source, source_dir, download=download, user_agent=user_agent)
        document_title = dataset_document_title(dataset_name, source)

        title_by_id[source.id] = document_title
        documents.append(
            {
                "id": source.id,
                "title": document_title,
                "path": local_path.relative_to(source_dir).as_posix(),
                "description": source.description,
                "status": "active",
                "acl": source.acl,
                "metadata": {
                    **source.metadata,
                    "source_url": source.url,
                    "source_file": local_path.name,
                },
            }
        )

        for index, question in enumerate(source.questions, start=1):
            case_facts = facts_for_question(source, index)
            evidence_aliases = evidence_aliases_for_question(source, index)
            cases.append(
                {
                    "case_name": f"{source.id}:qa:{index}",
                    "acting_user_email": acting_user_for_acl(source.acl),
                    "question": question,
                    "expected_document_ids": [source.id],
                    "expected_outcome": "answer",
                    "expected_key_facts": build_fact_specs(case_facts, document_title=document_title),
                    "expected_evidence_markers": build_fact_specs(
                        case_facts,
                        document_title=document_title,
                        extra_aliases=evidence_aliases,
                    ),
                    "scoring_notes": "真实中文企业/政务文档问答样例，检查中文检索、引用和关键事实覆盖。",
                    "metadata": {
                        **source.metadata,
                        "source_id": source.id,
                    },
                }
            )

        if source.questions and requires_manager(source.acl):
            cases.append(
                {
                    "case_name": f"{source.id}:permission:viewer_denied",
                    "acting_user_email": "viewer@local.test",
                    "question": source.questions[0],
                    "expected_document_ids": [],
                    "forbidden_document_ids": [source.id],
                    "expected_outcome": "refuse",
                    "forbidden_key_facts": build_fact_specs(source.expected_facts[:1], document_title=document_title),
                    "scoring_notes": "权限隔离样例：普通 viewer 不应看到 manager 角色文档或答案事实。",
                    "metadata": {
                        **source.metadata,
                        "source_id": source.id,
                        "permission_variant": "viewer_denied",
                    },
                }
            )

    cases.extend(build_composite_cases(COMPOSITE_CASES, source_by_id=source_by_id, title_by_id=title_by_id))
    cases.extend(build_composite_cases(LOW_OVERLAP_SCENARIO_CASES, source_by_id=source_by_id, title_by_id=title_by_id))

    return {
        "dataset_name": dataset_name,
        "documents": documents,
        "cases": cases,
        "metadata": {
            "language": "zh",
            "primary_profiles": [
                "zh_enterprise_ingestion",
                "zh_enterprise_retrieval",
                "zh_enterprise_qa",
                "zh_permission_isolation",
            ],
            "note": "真实公开中文企业/政务文档；OCR 不混入主效果分。",
        },
    }


def source_filename(source: SourceDocument) -> str:
    suffix = Path(urlparse(source.url).path).suffix.lower()
    if suffix not in {".pdf", ".html", ".htm"}:
        suffix = ".html"
    return f"{source.id}{suffix}"


def dataset_document_title(dataset_name: str, source: SourceDocument) -> str:
    domain = str(source.metadata.get("domain") or "doc").strip() or "doc"
    short_title = source.title.rsplit(":", 1)[-1].strip() or source.id
    return f"{dataset_name}:{domain}:{short_title}"


def facts_for_question(source: SourceDocument, question_index: int) -> list[str]:
    fact_index = question_index - 1
    if 0 <= fact_index < len(source.expected_facts):
        return [source.expected_facts[fact_index]]
    return source.expected_facts[:1]


def evidence_aliases_for_question(source: SourceDocument, question_index: int) -> list[str]:
    fact_index = question_index - 1
    aliases = source.evidence_aliases or []
    if 0 <= fact_index < len(aliases):
        return aliases[fact_index]
    return []


def build_composite_cases(
    composite_cases: list[CompositeCase],
    *,
    source_by_id: dict[str, SourceDocument],
    title_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in composite_cases:
        expected_source_ids = list(dict.fromkeys(source_id for source_id, _ in item.source_fact_indexes))
        if any(source_id not in source_by_id for source_id in expected_source_ids):
            continue
        expected_facts: list[dict[str, Any]] = []
        for source_id, fact_index in item.source_fact_indexes:
            source = source_by_id[source_id]
            if not 0 <= fact_index < len(source.expected_facts):
                raise IndexError(f"{item.case_name} references invalid fact index {source_id}:{fact_index}")
            expected_facts.extend(
                build_fact_specs(
                    [source.expected_facts[fact_index]],
                    document_title=title_by_id[source_id],
                    extra_aliases=evidence_aliases_for_question(source, fact_index + 1),
                )
            )
        cases.append(
            {
                "case_name": item.case_name,
                "acting_user_email": item.acting_user_email,
                "question": item.question,
                "expected_document_ids": expected_source_ids,
                "expected_outcome": "answer",
                "expected_key_facts": expected_facts,
                "expected_evidence_markers": expected_facts,
                "scoring_notes": item.scoring_notes,
                "metadata": item.metadata,
            }
        )
    return cases


def prepare_source_file(source: SourceDocument, source_dir: Path, *, download: bool, user_agent: str) -> Path:
    if is_html_source(source):
        clean_path = source_dir / f"{source.id}.md"
        raw_path = raw_html_source_path(source_dir, source)
        legacy_raw_path = source_dir / source_filename(source)
        if download:
            download_file(source.url, raw_path, user_agent=user_agent)
        elif not raw_path.exists() and legacy_raw_path.exists():
            raw_path = legacy_raw_path
        elif not raw_path.exists() and clean_path.exists() and clean_path.stat().st_size > 0:
            return clean_path
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing downloaded source for {source.id}: {raw_path}")
        clean_path.write_text(clean_html_document(source, raw_path), encoding="utf-8")
        return clean_path

    local_path = source_dir / source_filename(source)
    if download:
        download_file(source.url, local_path, user_agent=user_agent)
    if not local_path.exists() or local_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing downloaded source for {source.id}: {local_path}")
    return local_path


def is_html_source(source: SourceDocument) -> bool:
    suffix = Path(urlparse(source.url).path).suffix.lower()
    return suffix in {"", ".html", ".htm"}


def raw_html_source_path(source_dir: Path, source: SourceDocument) -> Path:
    return source_dir / "raw" / source_filename(source)


def download_file(url: str, target: Path, *, user_agent: str) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"Downloaded empty response from {url}")
    target.write_bytes(data)


class _VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in HTML_SKIP_TAGS:
            self._skip_depth += 1
            return
        if normalized in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in HTML_SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if normalized in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return "\n".join(self.parts)


def clean_html_document(source: SourceDocument, raw_path: Path) -> str:
    html = decode_html(raw_path.read_bytes())
    extractor = _VisibleTextExtractor()
    extractor.feed(html)
    short_title = source.title.rsplit(":", 1)[-1]
    lines = clean_visible_lines(extractor.text().splitlines(), title=short_title)
    body = "\n\n".join(lines)
    metadata_lines = [
        f"# {short_title}",
        "",
        f"来源机构：{source.metadata.get('source_org', '')}",
        f"原始链接：{source.url}",
    ]
    if body:
        metadata_lines.extend(["", body])
    return "\n".join(metadata_lines).rstrip() + "\n"


def decode_html(data: bytes) -> str:
    sample = data[:4096].decode("ascii", errors="ignore")
    match = re.search(r"charset=[\"']?([a-zA-Z0-9_-]+)", sample, flags=re.IGNORECASE)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "gb18030"])
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="ignore")


def clean_visible_lines(lines: list[str], *, title: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    title_key = compact_line(title)
    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = re.sub(r"^[|·•\-\s]+", "", line).strip()
        if not line or looks_like_ui_noise(line):
            continue
        key = compact_line(line)
        if not key or key == title_key or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return trim_to_document_body(cleaned)


def trim_to_document_body(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        compact = compact_line(line)
        if re.match(r"^第[一二三四五六七八九十百千万零〇两\d]+章", line):
            return trim_footer(lines[index:])
        if re.match(r"^第[一二三四五六七八九十百千万零〇两\d]+条", line):
            return trim_footer(lines[index:])
        if compact in {"中华人民共和国国务院令", "国家互联网信息办公室令"}:
            return trim_footer(lines[index:])
        if compact.endswith("令") and len(compact) <= 24 and any(token in compact for token in ("国务院", "办公室", "委员会", "部门")):
            return trim_footer(lines[index:])
    return trim_footer(lines)


def trim_footer(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if compact_line(line) in FOOTER_START_KEYS:
            return lines[:index]
    return lines


def looks_like_ui_noise(line: str) -> bool:
    compact = compact_line(line)
    if compact in UI_NOISE_EXACT:
        return True
    if any(pattern in line for pattern in UI_NOISE_PATTERNS):
        return True
    if len(compact) <= 8 and any(token in compact for token in ("登录", "注册", "分享", "打印", "收藏")):
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", line))
    if cjk_count <= 2 and re.search(r"https?://|www\.|\.gov|\.cn", line, flags=re.IGNORECASE):
        return True
    return False


def compact_line(value: str) -> str:
    return re.sub(r"[\s，。；：、,.!?！？;:()（）\[\]【】\"'“”‘’]+", "", value).casefold()


def acting_user_for_acl(acl: list[dict[str, Any]]) -> str:
    return "manager@local.test" if requires_manager(acl) else "viewer@local.test"


def requires_manager(acl: list[dict[str, Any]]) -> bool:
    return any(item.get("principal_type") == "role" and item.get("role_name") == "manager" for item in acl)


def build_fact_specs(
    facts: list[str],
    *,
    document_title: str | None = None,
    extra_aliases: list[str] | None = None,
) -> list[dict[str, Any]]:
    specs = []
    for fact in facts:
        aliases = [fact, relaxed_fact_alias(fact), *(extra_aliases or [])]
        spec: dict[str, Any] = {
            "label": fact,
            "aliases": dedupe(aliases),
            "weight": 1.0,
        }
        if document_title:
            spec["document_title"] = document_title.lower()
        specs.append(spec)
    return specs


def relaxed_fact_alias(fact: str) -> str:
    return re.sub(r"[，。、；：、“”‘’（）()\s]+", "", fact)


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result


if __name__ == "__main__":
    main()
