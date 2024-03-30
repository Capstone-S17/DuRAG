from src.generator import Generator
from tqdm import tqdm
from src.rds import db
import json
from tenacity import retry, stop_after_attempt, wait_exponential

with db.get_cursor() as cur:
    cur.execute("""SELECT extracted_page,pdf_document_name FROM "EXTRACTED_PDF_PAGE" """)
    pages = cur.fetchall()
curr_pdf = pages[0][1]
d = {curr_pdf:[]}
for text,pdf_name in pages:
    if pdf_name == curr_pdf:
        d[pdf_name].append(text)
    else:
        d[pdf_name] = [text]
        curr_pdf = pdf_name
with open('pdf_names.json',mode='r') as f:
    data = json.load(f)
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
def generate_question(splits,pdf_name):
    company = data[pdf_name]['company']
    oneshot =  '''###INSTRUCTION :Given that this is an annual report about Mooreast Holdings Ltd., generate a question base on the content of the 3 pages in this pdf. 
###CONTEXT: As at 30 The gross proceeds of the Listing amounted to approximately S$8.5 million (the “Gross June 2023 a total amount of approximately S$5.5 million out of the Gross Proceeds had been utilized according to the allocation set out in the Offer Document and the remaining balance of S$3.0 million is expected to be utilized as intended. Proceeds”).

Use of net proceeds | Amount   allocated   S$’000 | Balance as at   12 Apr 2023   S$’000 | Amount   utilised   S$’000 | Balance as at the   date of this   announcement   S$’000

Develop   and   grow | our |  |  |  | 
Renewable Division |  | 500 | 169 | – | 169
Development of facilities

and   capacity   of   51   Shipyard Road and scale   up of operations & invest   in plant, machineries &   equipment | 4,000 | 1,406 | – | 1,406

To explore opportunities   in M&A & strategic   alliances | 1,000 | 1,000 | – | 1,000
General corporate and

working capital purposes | 1,243 | 738 | (358) | (1) | 380
IPO expenses pursuant to

listing | 1,804 | – | – | –
Gross proceeds from the

Invitation | 8,547 | 3,313 | (358) | 2,955
Note:
(1) Approximately S$0.3 million was utilised as working capital of the Company to pay its ongoing professional expenses, directors’ remuneration and other corporate and administrative expenses.
The Company will continue to make periodic announcements on the utilisation of the remaining proceeds as and when such balance of the proceeds is materially disbursed.
The Company has received S$10 million from EDBI, which is currently placed in fixed deposit and pending deployment of fund. The Company will also utilise the proceeds in accordance with EDBI Notes Subscription Agreement.
BY ORDER OF THE BOARD

Sim Koon Lam
Chief Executive Officer
14 August 2023
###QUESTION: What is gross proceeds of the Listing amounted of Mooreast Holdings Ltd. 
###ANSWER: The gross proceeds of the Listing for Mooreast Holdings Ltd. amounted to approximately S$8.5 million.
###INSTRUCTION: Given that this is an annual report about LREIT., generate a question base on the content of the 3 pages in this pdf. 
###CONTEXT: Interested Person Transactions
The payments of the Manager’s management fees and acquisition fees, payments of property management fees, leasing fees and reimbursements to Lendlease Retail Pte. Ltd. (the “Property Manager”) in respect of payroll and related expenses, payment of management fees to Lendlease Italy SGR S.p.A. as well as payments of the Trustee’s fees and reimbursements pursuant to the Trust Deed are deemed to have been specifically approved by the Unitholders upon subscription for the Units on the listing of Lendlease Global Commercial REIT (“LREIT”) on 2 October 2019, and are therefore not subject to Rules 905 and 906 of the Listing Manual. Such payments are not to be included in the aggregate value of total interested person transactions as governed by Rules 905 and 906 of the Listing Manual.
Save as disclosed above, there were no other interested person transactions (excluding transactions less than S$100,000 each) entered into during FY2023 nor any material contracts entered into by LREIT that involved the interests of the Chief Executive Officer, any Director or controlling Unitholder of LREIT. Please also see significant related party transactions in Note 29 to the financial statements.
ISSUANCE OF LREIT UNITS
During FY2023, LREIT issued:
(i) an aggregate of 11,029,128 new Units (“Management Base Fee Units”) amounting to S$8.4 million as payment for the base fee element of the Manager’s management base fees;
(ii) an aggregate of 4,330,102 new Units (“Management Performance Fee Units”) amounting to S$3.4 million as payment for the performance fee element of the Manager’s management performance fees;
(iii) an aggregate of 5,463,895 new Units (“Property Management Fee Units”) amounting to S$4.1 million as payment for the Property Manager’s management fees; and
(iv) an aggregate of 25,712,783 new Units amounting to S$17.5 million pursuant to LREIT’s Distribution Reinvestment Plan in respect of the distribution of 2.4499 cents per Unit for the period from 1 July 2022 to 31 December 2022.
Lendlease GCR Investment Holding Pte. Ltd. has been nominated by each of the Manager and the Property Manager to receive the Management Base Fee Units, Management Performance Fee Units and the Property Management Fee Units in accordance with the Trust Deed and the master property management agreement relating to the properties of LREIT respectively.
Statistics of Unitholdings
As at 7 September 2023
ISSUED AND FULLY PAID UNITS
2,323,661,727 Units issued in LREIT as at 7 September 2023 (voting rights: 1 vote per unit).
There is only one class of Units in LREIT. There are no treasury units and no subsidiary holdings held.
DISTRIBUTION OF UNITHOLDINGS
Size of Unitholdings
1 – 99
100 – 1,000
1,001 – 10,000
10,001 – 1,000,000
1,000,001 and above
Total
TWENTY LARGEST UNITHOLDERS
No. Name
1 DBS VICKERS SECURITIES (SINGAPORE) PTE. LTD.
2 CITIBANK NOMINEES SINGAPORE PTE. LTD.
3 DBS NOMINEES (PRIVATE) LIMITED
4 HSBC (SINGAPORE) NOMINEES PTE. LTD.
5 RAFFLES NOMINEES (PTE.) LIMITED
6 DBSN SERVICES PTE. LTD.
7 UNITED OVERSEAS BANK NOMINEES (PRIVATE) LIMITED
8 HPL INVESTERS PTE. LTD.
9 BPSS NOMINEES SINGAPORE (PTE.) LTD.
10 IFAST FINANCIAL PTE. LTD.
11 PHILLIP SECURITIES PTE. LTD.
12 DB NOMINEES (SINGAPORE) PTE. LTD.
13 TIGER BROKERS (SINGAPORE) PTE. LTD.
14 CGS-CIMB SECURITIES (SINGAPORE) PTE. LTD.
15 UOB KAY HIAN PRIVATE LIMITED
16 OCBC NOMINEES SINGAPORE PRIVATE LIMITED
17 MAYBANK SECURITIES PTE. LTD.
18 OCBC SECURITIES PRIVATE LIMITED
19 ABN AMRO CLEARING BANK N.V.
20 MORGAN STANLEY ASIA (SINGAPORE) SECURITIES PTE. LTD.
Total
No. of Unitholders
29
998
8,310
5,697
40
15,074
% No. of Units
0.19 968
6.62 852,985
55.13 41,763,850
37.79 255,789,959
0.27 2,025,253,965
100.00 2,323,661,727
No. of Units
642,195,127
313,190,576
299,930,606
205,946,876
174,551,377
131,990,996
33,780,529
29,238,753
27,586,724
27,320,323
19,232,854
14,709,367
9,916,268
9,719,301
9,444,170
9,364,077
7,909,304
6,825,954
6,459,118
5,707,351
1,985,019,651
%
0.00
0.03
1.80
11.01
87.16
100.00
%
27.64
13.48
12.91
8.86
7.51
5.68
1.45
1.26
1.19
1.18
0.83
0.63
0.43
0.42
0.41
0.40
0.34
0.29
0.28
0.25
85.44
###QUESTION: What is the largest unitholder of LREIT as of 7 September 2023?
###ANSWER: DBS VICKERS SECURITIES (SINGAPORE) PTE. LTD.'''
    instruction = f'''###INSTRUCTION: Given that this is an annual report about {company}, generate a question and answer base on the content of the 3 pages in this pdf. Your question should have the same difficulty as the example.'''
    prompt = f'''{oneshot}\n{instruction}\n'''+"###CONTEXT:"+"\n\n".join(splits)+"\n###QUESTION:"
    
    out = gemini_pipe(prompt)
    return out
    
def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False)
    f.write('\n')
    f.close()
g = Generator()
def gemini_pipe(prompt):
    out = g.gemini.generate_content(prompt,stream=False)
    return out
    
from tqdm import tqdm
output = []
k = {}
for key in tqdm(d.keys()):
    page_list = d[key]
    idx = 0
    while idx<len(page_list):
        splits = page_list[idx:idx+3]
        out = generate_question(splits,key)
        idx += 3
        try:
            question,answer = out.text.split('###ANSWER:')
        except:
            continue
        output = {"question":question.strip(),
                  "answer":answer.strip(),
                  "raw_text":out.text,
                  "pages":splits,
                  "pdf_name":key,
                  "page_number":(idx,idx+3)
                 }
        print(question,answer)
        write_json(output,'generated_question_v2.json')
