general_legal_chat_dataset = [
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm starting a small consulting business and want to know the difference between forming an LLC versus incorporating. What are the key considerations for tax treatment, liability protection, and operational flexibility?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "For a small consulting business, both LLCs and corporations offer liability protection, but they differ significantly in structure and tax treatment. LLC advantages: (1) Tax flexibility - can elect S-corp, partnership, or sole proprietorship taxation; (2) Operational simplicity - fewer formalities, no board requirements; (3) Pass-through taxation avoids double taxation; (4) Flexible profit/loss distribution among members. Corporation advantages: (1) Well-established legal framework with extensive precedent; (2) Better for raising capital and bringing investors; (3) Potential tax benefits for retained earnings; (4) Stock options for employee compensation. Key considerations: (1) Self-employment taxes - LLC members pay on all income, S-corp shareholders only on salary portion; (2) Growth plans - corporations better for expansion and going public; (3) Operational complexity - corporations require boards, shareholder meetings, formal records; (4) State-specific factors - some states favor LLCs, others corporations. For most small consulting businesses, LLCs provide optimal balance of protection, simplicity, and tax efficiency. However, if you plan rapid growth or bringing investors, incorporation might be preferable. Consult a business attorney and tax professional for your specific situation.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 1,
            "difficulty": "beginner",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My neighbor's tree fell and damaged my fence during a storm. Am I responsible for the repair costs, or is my neighbor liable? There was no negligence involved - it was just a natural weather event."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Tree damage liability depends on several factors, and the answer varies by state law. Generally, if a healthy tree falls due to natural causes (storms, wind) without negligence, the property owner where the damage occurred bears the loss under the 'act of God' doctrine. However, liability can shift if: (1) The tree owner knew or should have known the tree was diseased, dead, or unstable; (2) Previous complaints about the tree's condition were ignored; (3) The tree was obviously hazardous to a reasonable person. Many states follow the 'Massachusetts Rule' where tree owners are liable only for negligent maintenance, while others apply general negligence principles. Key factors courts consider: (1) Tree's visible condition before the incident; (2) Whether a reasonable inspection would have revealed problems; (3) Prior notice of dangerous conditions; (4) Local ordinances about tree maintenance. Practical steps: (1) Document the damage with photos; (2) Check if the tree showed obvious signs of disease or damage; (3) Review your homeowner's insurance coverage; (4) Consult local property law since rules vary significantly by jurisdiction. If negligence isn't involved, you'll likely need to repair the fence yourself or through insurance, but consultation with a local attorney can clarify your state's specific rules.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 2,
            "difficulty": "beginner",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I received a cease and desist letter claiming my blog post infringes someone's trademark. The letter demands I take down the post and pay damages. How should I evaluate whether their claim has merit?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Evaluating a trademark cease and desist requires analyzing several key factors. First, determine if valid trademark rights exist: (1) Is the mark federally registered or used in commerce? (2) What goods/services does it cover? (3) Is it distinctive enough for protection? Next, assess likelihood of confusion using factors like: (1) Similarity of marks in appearance, sound, meaning; (2) Similarity of goods/services; (3) Channels of trade and consumer sophistication; (4) Strength of the trademark. Important defenses to consider: (1) Fair use - using mark descriptively or for commentary; (2) Nominative fair use - referring to trademark owner's products; (3) First Amendment protection for editorial/journalistic use; (4) Lack of commercial use if blog is non-commercial. Red flags in the letter: (1) Vague claims without specific legal basis; (2) Demands for excessive damages; (3) Threats without explaining actual harm; (4) Short deadlines designed to pressure quick settlement. Recommended steps: (1) Don't ignore the letter but don't panic; (2) Preserve all evidence including your blog post; (3) Research the claiming party's actual trademark rights; (4) Consider whether your use could realistically confuse consumers; (5) Consult an intellectual property attorney for complex cases. Many cease and desist letters are overreaching attempts to suppress legitimate speech.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "intellectual_property"
        },
        "extra_info": {
            "split": "train",
            "index": 3,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering a prenuptial agreement before marriage. What can and cannot be included in a prenup, and what makes them enforceable? Are there any major red flags to avoid?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Prenuptial agreements can address many financial matters but have important limitations. Typically enforceable provisions include: (1) Property division - separate vs. marital property designation; (2) Spousal support/alimony waiver or limitation; (3) Inheritance rights and estate planning coordination; (4) Business ownership protection; (5) Debt responsibility allocation; (6) Financial decision-making during marriage. Prohibited or unenforceable provisions: (1) Child custody or support - courts retain jurisdiction over children's best interests; (2) Illegal activities or provisions violating public policy; (3) Domestic duties or personal behavior requirements; (4) Provisions that leave one spouse impoverished. Enforceability requirements vary by state but generally include: (1) Voluntary execution without duress or coercion; (2) Full financial disclosure by both parties; (3) Fair and reasonable terms (procedural and substantive); (4) Proper legal formalities - written, signed, witnessed/notarized; (5) Independent legal representation recommended. Red flags that void prenups: (1) Signing shortly before wedding under pressure; (2) Hiding assets or debts; (3) Extremely one-sided terms; (4) No opportunity to review with attorney; (5) Misrepresentation of financial circumstances. Best practices: (1) Start discussions early; (2) Full financial transparency; (3) Separate attorneys for each party; (4) Regular review and potential updates; (5) Focus on protection rather than punishment.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 4,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My employer wants me to sign a non-compete agreement for my marketing position. It prohibits working for competitors for 2 years within 50 miles. Is this enforceable, and what should I negotiate?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Non-compete enforceability varies dramatically by state, with some states (like California) banning them entirely while others enforce reasonable restrictions. Key enforceability factors include: (1) Legitimate business interest protection (trade secrets, customer relationships, specialized training); (2) Reasonable geographic scope relative to business operations; (3) Reasonable time duration; (4) Adequate consideration (new job, promotion, access to confidential information); (5) Narrow scope that doesn't prevent earning a living. Your agreement's 2-year/50-mile restriction may be problematic because: (1) Marketing roles often involve transferable skills rather than trade secrets; (2) 2 years is lengthy for most positions; (3) 50 miles could encompass multiple job markets; (4) Must be necessary to protect legitimate interests. Negotiation strategies: (1) Narrow the scope to specific clients you directly serve; (2) Reduce time period to 6-12 months; (3) Limit geographic restriction to actual competition area; (4) Include garden leave clause (continued salary during restriction); (5) Add carve-outs for industry standard practices. Alternative approaches: (1) Non-solicitation of specific clients instead of broad competition ban; (2) Confidentiality agreement protecting actual trade secrets; (3) Liquidated damages clause rather than injunctive relief. Red flags: (1) Applying to all employees regardless of access to sensitive information; (2) Restricting fundamental right to work; (3) No additional consideration provided. Consider consulting employment attorney familiar with your state's laws.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 5,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to leave my estate to animal charities but my family is threatening to contest the will. What makes a will legally valid and how can I prevent successful challenges?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "A valid will requires: (1) Legal capacity - being of sound mind and over 18; (2) Testamentary intent - clear intention to dispose of property at death; (3) Proper execution - written, signed by testator, witnessed according to state law; (4) Freedom from undue influence, fraud, or duress. To prevent successful challenges to your charitable will: (1) Use experienced estate attorney to ensure proper execution; (2) Document your mental capacity with medical records showing competency; (3) Include statement explaining your charitable intentions and family circumstances; (4) Consider no-contest clause penalizing challengers; (5) Make modest family bequests to show you considered but chose charity over family; (6) Use video will-signing to demonstrate capacity and voluntary nature. Common challenge grounds include: (1) Lack of capacity due to age, illness, or medication; (2) Undue influence by charities or advisors; (3) Failure to follow state formalities; (4) Fraud in will preparation. Protective measures: (1) Annual capacity assessments if health declining; (2) Independent legal representation without charity involvement; (3) Multiple witnesses to signing ceremony; (4) Keep detailed records of decision-making process. Some states have enhanced protection for charitable bequests. Consider revocable trust as alternative - harder to challenge and provides privacy. Document your reasoning and maintain evidence of independent decision-making throughout the process.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "estate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 6,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I bought a used car from a dealer and discovered it has frame damage that wasn't disclosed. The dealer now refuses to take it back. What are my legal options under consumer protection laws?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "You have several potential legal remedies for undisclosed frame damage. Consumer protection options include: (1) State lemon laws - may apply to used cars if dealer provides warranty; (2) Unfair and Deceptive Acts and Practices (UDAP) laws - prohibit misleading sales practices; (3) Implied warranty of merchantability - vehicle must be fit for ordinary use; (4) Magnuson-Moss Warranty Act - federal protection for written warranties. Key evidence to gather: (1) All sales documents and advertisements; (2) Vehicle history reports; (3) Independent mechanic inspection documenting damage; (4) Photos of damage and repair evidence; (5) Communications with dealer about the issue. Potential remedies: (1) Rescission - cancel sale and return vehicle for refund; (2) Damages - repair costs and diminished value; (3) Revocation of acceptance under UCC if substantially impaired; (4) Breach of warranty claims; (5) Fraud claims if dealer knew and concealed damage. Steps to take: (1) Document everything immediately; (2) Get professional inspection and estimate; (3) Send written demand letter to dealer; (4) File complaint with state consumer protection agency; (5) Contact state attorney general's office; (6) Consider small claims court for smaller amounts. Time limits are crucial - most states have statutes of limitations for these claims. Many states have specific used car disclosure requirements that may strengthen your case. Consider consulting consumer protection attorney for significant damages.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "consumer_protection"
        },
        "extra_info": {
            "split": "train",
            "index": 7,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My landlord wants to raise my rent by 40% with only 30 days notice. I have a month-to-month lease. Is this legal, and what are my rights as a tenant?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Rent increase rules vary significantly by state and local jurisdiction. For month-to-month tenancies, key considerations include: (1) Notice requirements - most states require 30 days for monthly tenancies, but some require more; (2) Rent control laws - some cities limit increase amounts and frequency; (3) Retaliatory increase prohibitions - can't raise rent to punish tenant for complaints; (4) Discrimination protections - increases can't be based on protected characteristics. A 40% increase raises red flags for: (1) Possible retaliation if you recently complained about conditions; (2) Rent control violations in regulated areas; (3) Discriminatory intent; (4) Attempt to constructively evict. Your options may include: (1) Negotiating a smaller increase; (2) Challenging the increase if rent control applies; (3) Filing complaint if retaliation suspected; (4) Seeking legal aid for discrimination claims; (5) Looking for new housing if increase is lawful. Research needed: (1) Check local rent control ordinances; (2) Review your lease for any restrictions; (3) Document any recent complaints or issues; (4) Compare to market rates for similar units; (5) Check state tenant rights laws. Important deadlines: (1) You typically must respond before the increase takes effect; (2) Challenge periods may be short; (3) Moving requires proper notice to landlord. Contact local tenant rights organizations or legal aid for jurisdiction-specific advice. Many areas have tenant hotlines for immediate guidance.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "real_estate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 8,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I was in a car accident and the other driver was clearly at fault, but their insurance company is offering a settlement that seems too low. Should I accept it or hire a lawyer?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Before accepting any settlement, carefully evaluate whether it covers all your damages. Recoverable damages typically include: (1) Medical expenses (current and future); (2) Lost wages and earning capacity; (3) Property damage; (4) Pain and suffering; (5) Emotional distress; (6) Loss of consortium (if married). Red flags suggesting low settlement: (1) Offer made before you've finished medical treatment; (2) Doesn't cover all medical bills; (3) No compensation for lost wages; (4) Minimal pain and suffering amount; (5) Quick pressure to accept. Factors favoring attorney consultation: (1) Significant injuries requiring ongoing treatment; (2) Disputed liability or comparative negligence; (3) Complex medical issues; (4) Lost wages or earning capacity; (5) Insurance company acting in bad faith; (6) Settlement under $10,000 (may handle yourself) vs. over $25,000 (likely need attorney). Steps before deciding: (1) Get all medical records and bills; (2) Calculate total economic damages; (3) Research similar case settlements; (4) Understand your state's statute of limitations; (5) Consider future medical needs. Attorney benefits: (1) Experience valuing claims; (2) Negotiation skills; (3) Litigation threat; (4) Contingency fee basis (usually 33-40%); (5) Resources for case development. You can also try negotiating yourself first, keeping attorney option open. Document everything and don't sign anything until you're confident about the full extent of your damages.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "personal_injury"
        },
        "extra_info": {
            "split": "train",
            "index": 9,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm getting divorced and we can't agree on child custody. What factors do courts consider when determining custody arrangements, and how can I strengthen my case?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Courts determine custody based on the 'best interests of the child' standard, considering numerous factors: (1) Child's physical, emotional, and developmental needs; (2) Each parent's ability to provide stable, loving environment; (3) Existing parent-child relationships and bonds; (4) Continuity and stability of current arrangements; (5) Parents' mental and physical health; (6) History of domestic violence or substance abuse; (7) Child's preference (if age-appropriate); (8) Parents' willingness to facilitate relationship with other parent; (9) Geographic proximity and practical considerations; (10) Work schedules and childcare arrangements. To strengthen your case: (1) Document your involvement in child's daily life (school, activities, medical care); (2) Maintain detailed records of time spent with child; (3) Show stability in housing, employment, and lifestyle; (4) Demonstrate support for child's relationship with other parent; (5) Keep records of any concerning behavior by other parent; (6) Get involved in child's school and activities; (7) Maintain consistent parenting schedule during separation. What courts don't want to see: (1) Attempts to alienate child from other parent; (2) Using child as messenger or spy; (3) Instability in housing or employment; (4) New romantic relationships disrupting child's routine; (5) Substance abuse or criminal behavior. Consider mediation before litigation - it's often faster, cheaper, and less traumatic for children. Many states require parenting classes during divorce proceedings. Focus on your child's needs rather than grievances against your ex-spouse.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 10,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I was arrested for DUI but I believe the traffic stop was illegal. The officer said I was swerving but I wasn't. What should I know about challenging the stop and the charges?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DUI cases have multiple potential defense strategies, and challenging the initial stop is one approach. Fourth Amendment requires reasonable suspicion for traffic stops. Common challenges include: (1) Lack of reasonable suspicion - officer must articulate specific, articulable facts suggesting traffic violation or criminal activity; (2) Pretextual stops - using minor traffic violations to investigate unrelated crimes; (3) Improper field sobriety tests - must be conducted according to standardized procedures; (4) Breathalyzer issues - machine calibration, maintenance, operator certification; (5) Chain of custody problems with blood tests. Key evidence to examine: (1) Police report and dash cam footage; (2) Officer's stated reason for stop; (3) Witness statements; (4) Your own recollection and any video; (5) Breathalyzer maintenance records; (6) Officer's training and certification. Potential consequences beyond criminal charges: (1) License suspension through DMV (separate from criminal case); (2) Ignition interlock device requirements; (3) Increased insurance rates; (4) Employment impacts; (5) Professional license issues. Important timelines: (1) DMV hearing request (usually 10 days); (2) Arraignment and plea deadlines; (3) Discovery deadlines; (4) Pretrial motions. DUI defense requires specialized knowledge of: (1) Scientific evidence and testing procedures; (2) Constitutional protections; (3) Local court practices; (4) Plea negotiation strategies. Given the complexity and serious consequences, strongly consider hiring an experienced DUI attorney. Many offer free consultations and can evaluate the strength of your case.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 11,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to trademark my small business name and logo. What's the process, how much does it cost, and what protection does it provide?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Trademark registration provides valuable business protection, but the process requires careful planning. Basic requirements: (1) Mark must be distinctive and used in commerce; (2) Cannot conflict with existing trademarks; (3) Must identify specific goods/services; (4) Cannot be merely descriptive or generic. The registration process involves: (1) Comprehensive trademark search to avoid conflicts; (2) Application filing with USPTO ($250-$350 per class online); (3) Examination by USPTO attorney (6-8 months); (4) Publication for opposition (30 days); (5) Registration certificate if no opposition. Total costs typically range: (1) DIY application: $250-$700; (2) With attorney: $1,500-$3,000; (3) Ongoing maintenance fees required. Trademark protection provides: (1) Exclusive right to use mark nationwide; (2) Legal presumption of ownership; (3) Ability to stop infringers; (4) Basis for international registration; (5) Enhanced damages for willful infringement; (6) Customs enforcement against counterfeits. Important considerations: (1) Conduct thorough search before filing; (2) Choose appropriate goods/services classifications; (3) Consider state registration for local businesses; (4) Maintain continuous use to keep rights; (5) File renewal documents on time. Common mistakes: (1) Choosing weak, descriptive marks; (2) Inadequate search leading to conflicts; (3) Wrong goods/services classification; (4) Abandoning applications due to office actions. Consider hiring trademark attorney for: (1) Complex applications; (2) Valuable brands; (3) Potential conflicts; (4) International plans. Start using ™ symbol immediately, ® only after registration.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "intellectual_property"
        },
        "extra_info": {
            "split": "train",
            "index": 12,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My employer fired me after I reported safety violations to OSHA. Can they do this, and what legal protections do I have as a whistleblower?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Federal and state laws provide strong whistleblower protections for reporting safety violations. Your situation likely violates multiple laws: (1) OSHA Section 11(c) - prohibits retaliation for reporting workplace safety hazards; (2) Sarbanes-Oxley Act - protects reports of corporate fraud; (3) False Claims Act - protects reports of government fraud; (4) State whistleblower statutes - vary by jurisdiction. Protected activities include: (1) Filing OSHA complaints; (2) Testifying about safety violations; (3) Refusing to work in dangerous conditions; (4) Reporting violations to supervisors; (5) Participating in safety inspections. To prove retaliation, you must show: (1) Engaged in protected activity; (2) Employer knew about the activity; (3) Suffered adverse employment action; (4) Causal connection between activity and adverse action. Remedies available: (1) Reinstatement to your job; (2) Back pay and benefits; (3) Compensatory damages; (4) Attorney fees; (5) Punitive damages in some cases. Critical deadlines: (1) OSHA complaint must be filed within 30 days of adverse action; (2) Other federal claims have 180-day deadlines; (3) State law deadlines vary. Immediate steps: (1) Document the retaliation thoroughly; (2) File OSHA complaint quickly; (3) Preserve all evidence; (4) Avoid signing separation agreements without attorney review; (5) Consider consulting employment attorney. Evidence to gather: (1) Documentation of safety reports; (2) Timing of adverse action; (3) Employer's stated reasons; (4) Witness statements; (5) Personnel records. Strong retaliation case given clear temporal connection between OSHA report and termination.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 13,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm buying my first house and the seller wants to close in 10 days. The inspection revealed some issues. What should I know about the closing process and my rights?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "A 10-day closing is extremely tight and requires careful attention to contingencies and deadlines. Key closing components: (1) Title search and insurance; (2) Final walkthrough; (3) Loan funding; (4) Document signing; (5) Funds transfer; (6) Deed recording. Your rights regarding inspection issues: (1) Review inspection contingency deadline in your contract; (2) Request repairs, credits, or price reduction; (3) Withdraw from purchase if issues are significant; (4) Negotiate with seller on resolution. Common inspection issues and responses: (1) Minor repairs - typically request seller fix; (2) Major systems (HVAC, electrical, plumbing) - may need credits or price reduction; (3) Structural issues - consider withdrawal or significant price adjustment; (4) Safety hazards - must be addressed before closing. Red flags requiring immediate attention: (1) Title issues or liens; (2) Loan problems or qualification changes; (3) Seller's inability to provide clear title; (4) Significant property damage between contract and closing. Essential pre-closing steps: (1) Final walkthrough within 24 hours; (2) Verify all agreed repairs completed; (3) Confirm loan terms haven't changed; (4) Review all closing documents; (5) Bring certified funds for closing costs. Common closing costs: (1) Loan origination fees; (2) Title insurance; (3) Attorney fees; (4) Recording fees; (5) Homeowner's insurance; (6) Property taxes; (7) HOA fees. Consider hiring real estate attorney for: (1) Contract review; (2) Title examination; (3) Closing representation; (4) Complex issues resolution. Don't rush - better to extend closing than make costly mistakes.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "real_estate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 14,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm being sued by a debt collector for a credit card debt I don't think I owe. What should I do and what are my rights under debt collection laws?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Debt collection lawsuits require immediate response to protect your rights. Key protections under federal and state law: (1) Fair Debt Collection Practices Act (FDCPA) - regulates debt collector conduct; (2) Fair Credit Reporting Act (FCRA) - governs credit reporting; (3) State consumer protection laws - vary by jurisdiction. Your rights include: (1) Demand validation of debt; (2) Dispute inaccurate information; (3) Request proof of debt ownership; (4) Challenge collector's legal standing; (5) Raise statute of limitations defense. Common defenses to debt collection lawsuits: (1) Statute of limitations expired; (2) Debt already paid; (3) Identity theft or fraud; (4) Incorrect amount claimed; (5) Collector lacks standing to sue; (6) Procedural defects in lawsuit. Immediate steps: (1) Don't ignore the lawsuit - respond within required timeframe; (2) Request debt validation if not already done; (3) Gather all documentation about the debt; (4) Check your credit reports; (5) Consider consulting consumer attorney. Response strategies: (1) File formal answer raising defenses; (2) Demand proof of debt ownership; (3) Challenge accuracy of amount; (4) Assert statute of limitations if applicable; (5) Counterclaim for FDCPA violations if appropriate. Evidence to gather: (1) Original credit card statements; (2) Payment records; (3) Correspondence with collector; (4) Documentation of identity theft; (5) Proof of debt dispute. Many debt collection lawsuits involve: (1) Insufficient documentation; (2) Expired statutes of limitations; (3) Incorrect debt amounts; (4) Lack of standing to sue. Consider that collectors often prefer settlement to trial costs.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "consumer_protection"
        },
        "extra_info": {
            "split": "train",
            "index": 15,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to file a complaint against my doctor for malpractice. What do I need to prove, and what's the process for pursuing a medical malpractice claim?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Medical malpractice cases are complex and require proving four essential elements: (1) Duty - doctor-patient relationship existed; (2) Breach - doctor violated standard of care; (3) Causation - breach caused your injury; (4) Damages - you suffered actual harm. Standard of care means treatment that a reasonably competent doctor would provide under similar circumstances. Expert testimony is typically required to establish: (1) What standard of care required; (2) How doctor's actions deviated from standard; (3) How deviation caused your injury; (4) Extent of damages. Common types of malpractice: (1) Misdiagnosis or delayed diagnosis; (2) Surgical errors; (3) Medication mistakes; (4) Birth injuries; (5) Failure to obtain informed consent; (6) Failure to follow up on test results. Process overview: (1) Obtain all medical records; (2) Consult medical expert for case evaluation; (3) File complaint within statute of limitations; (4) Discovery phase with depositions; (5) Expert witness testimony; (6) Settlement negotiations or trial. Critical deadlines: (1) Statute of limitations (typically 2-3 years from discovery); (2) Statute of repose (absolute deadline regardless of discovery); (3) Notice requirements in some states. Challenges in malpractice cases: (1) High burden of proof; (2) Expensive expert witnesses; (3) Medical records complexity; (4) Doctors' strong legal representation; (5) Jury sympathy for healthcare providers. Most malpractice attorneys work on contingency (typically 33-40% of recovery) and provide free case evaluations. Consider that: (1) Most cases settle before trial; (2) Success rates are relatively low; (3) Process can take years; (4) Emotional toll can be significant.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "medical_malpractice"
        },
        "extra_info": {
            "split": "train",
            "index": 16,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm a freelance graphic designer and a client is refusing to pay for completed work. What are my options for collecting payment and protecting my intellectual property?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Freelancers have several legal remedies for non-payment and IP protection. Payment collection options: (1) Demand letter with deadline; (2) Lien rights on work product; (3) Small claims court (typically under $10,000); (4) Civil lawsuit for larger amounts; (5) Debt collection agency; (6) Mechanic's lien in some states. Intellectual property protections: (1) Copyright automatically exists in your original work; (2) Trademark may protect logos/brand elements; (3) Work-for-hire agreements affect ownership; (4) Licensing terms control usage rights. Contract terms that strengthen your position: (1) Clear payment terms and deadlines; (2) Late payment penalties; (3) Retention of rights until payment; (4) Kill fee provisions; (5) Termination clauses. Steps to take immediately: (1) Document all work performed; (2) Preserve original files and communications; (3) Send formal demand letter; (4) Cease delivering additional work; (5) Register copyright if significant value. Key evidence to gather: (1) Written agreement or email communications; (2) Invoices and payment records; (3) Work samples and deliverables; (4) Client feedback and approval; (5) Time records and expenses. Legal strategies: (1) Quantum meruit (payment for work performed); (2) Breach of contract claim; (3) Copyright infringement if client uses unpaid work; (4) Unjust enrichment theory. Practical considerations: (1) Cost of legal action vs. amount owed; (2) Client's ability to pay; (3) Reputation and relationship factors; (4) Time and emotional investment. Prevention for future projects: (1) Detailed written contracts; (2) Partial payment schedules; (3) Clear scope and change order procedures; (4) Copyright and usage rights clauses.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 17,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I was denied a job and suspect it was because of my age (I'm 55). How can I prove age discrimination and what legal remedies are available?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Age discrimination in employment is prohibited by the Age Discrimination in Employment Act (ADEA) for workers 40 and older. To prove age discrimination, you must establish: (1) You're in protected class (40+); (2) You applied and were qualified for the position; (3) You were rejected despite qualifications; (4) Position remained open or went to younger person. Evidence of age discrimination: (1) Comments about age, retirement, or 'new blood'; (2) Patterns of hiring younger candidates; (3) Job postings seeking 'recent graduates' or 'digital natives'; (4) Statistical evidence of age bias; (5) Comparative treatment of younger applicants. Types of evidence to gather: (1) Job posting and application materials; (2) Interview notes and correspondence; (3) Company demographics and hiring patterns; (4) Witness statements about discriminatory comments; (5) Social media posts or internal communications. Legal process: (1) File EEOC complaint within 180 days (300 in some states); (2) EEOC investigation and mediation opportunity; (3) Right to sue letter if no resolution; (4) Federal court lawsuit within 90 days. Available remedies: (1) Hiring with back pay; (2) Promotion and salary adjustments; (3) Compensatory damages for emotional distress; (4) Punitive damages for willful violations; (5) Attorney fees and costs. Challenges in age discrimination cases: (1) Difficult to prove discriminatory intent; (2) Employers often have legitimate business reasons; (3) Circumstantial evidence typically required; (4) Burden of proof on plaintiff. Practical steps: (1) Document all interactions; (2) Keep detailed records of job search; (3) File EEOC complaint promptly; (4) Consider consulting employment attorney; (5) Continue job search while pursuing claim.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 18,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm starting a food truck business and need to understand the licensing requirements, liability issues, and legal structure. What should I know?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Food truck businesses involve complex regulatory compliance across multiple levels. Licensing requirements typically include: (1) Business license from city/county; (2) Food service license from health department; (3) Mobile vendor permit; (4) Fire department approval; (5) Zoning compliance for parking locations; (6) State sales tax permit. Health and safety regulations: (1) Commercial kitchen commissary requirements; (2) Food handler certifications; (3) Regular health inspections; (4) Waste disposal compliance; (5) Water and propane safety standards. Business structure considerations: (1) LLC provides liability protection and tax flexibility; (2) Corporation offers more formality for investors; (3) Sole proprietorship is simpler but offers no liability protection; (4) Partnership requires detailed agreements. Essential insurance coverage: (1) General liability insurance; (2) Commercial auto insurance; (3) Product liability coverage; (4) Workers' compensation if employees; (5) Business interruption insurance. Legal risks to address: (1) Food poisoning liability; (2) Vehicle accidents and injuries; (3) Permit violations and fines; (4) Employment law compliance; (5) Contract disputes with vendors. Location and zoning issues: (1) Private property permission; (2) Public space permits; (3) Distance requirements from restaurants; (4) Time restrictions and curfews; (5) Special event permits. Financial and tax considerations: (1) Sales tax collection and remittance; (2) Employment taxes if hiring; (3) Business expense deductions; (4) Quarterly estimated taxes; (5) Record-keeping requirements. Start with: (1) Research local regulations thoroughly; (2) Consult small business attorney; (3) Obtain proper business structure; (4) Secure comprehensive insurance; (5) Develop compliance procedures.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 19,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My teenage daughter was arrested for shoplifting and the store is demanding payment for damages. What are the legal consequences and how does juvenile court work?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Juvenile shoplifting cases involve both criminal and civil consequences. Criminal process: (1) Arrest and booking; (2) Detention hearing if held; (3) Intake and diversion options; (4) Petition filing if case proceeds; (5) Adjudication hearing (juvenile trial); (6) Disposition (sentencing). Potential criminal consequences: (1) Probation and community service; (2) Restitution to store; (3) Counseling or classes; (4) Juvenile detention; (5) Juvenile record (may be sealable). Civil demand letter: (1) Stores can demand civil penalties beyond criminal case; (2) Typically $50-$500 plus merchandise value; (3) Not required to pay without lawsuit; (4) Consider negotiation or challenge if excessive. Juvenile court differences: (1) Focus on rehabilitation, not punishment; (2) Closed proceedings for privacy; (3) Parents typically required to attend; (4) Different terminology (adjudication vs. conviction); (5) Records may be sealed or expunged. Strategies for defense: (1) Negotiate diversion program; (2) Argue first offense and good character; (3) Demonstrate remorse and responsibility; (4) Show family support and supervision; (5) Propose community service or restitution. Long-term considerations: (1) Impact on college applications; (2) Employment background checks; (3) Record sealing eligibility; (4) Repeat offense consequences; (5) Learning opportunity for responsibility. Important steps: (1) Hire juvenile defense attorney; (2) Don't discuss case without attorney; (3) Gather character references; (4) Consider counseling proactively; (5) Address underlying issues. Civil demand response: (1) Review demand letter carefully; (2) Verify actual damages claimed; (3) Consider negotiation; (4) Don't ignore completely; (5) Consult attorney if amount is significant.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 20,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm concerned about protecting my digital privacy and personal data. What legal rights do I have regarding companies collecting and using my information?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Digital privacy rights vary by jurisdiction and are rapidly evolving. Current legal protections include: (1) State privacy laws (California CCPA, Virginia CDPA, etc.); (2) Sector-specific federal laws (HIPAA, FERPA, FCRA); (3) FTC enforcement of unfair/deceptive practices; (4) Contract law through terms of service; (5) Constitutional protections in limited circumstances. California Consumer Privacy Act (CCPA) provides: (1) Right to know what data is collected; (2) Right to delete personal information; (3) Right to opt-out of data sales; (4) Right to non-discrimination for exercising rights; (5) Private right of action for data breaches. Your rights under various laws: (1) Access to your data; (2) Correction of inaccurate information; (3) Deletion of unnecessary data; (4) Portability of your data; (5) Opt-out of certain processing. Steps to protect your privacy: (1) Review and adjust privacy settings; (2) Read privacy policies of services you use; (3) Use strong, unique passwords; (4) Enable two-factor authentication; (5) Be selective about data sharing. Legal remedies for privacy violations: (1) State attorney general enforcement; (2) Class action lawsuits; (3) Individual lawsuits under state laws; (4) FTC complaints; (5) Industry-specific regulators. Practical privacy strategies: (1) Use privacy-focused browsers and search engines; (2) Limit social media data sharing; (3) Review app permissions regularly; (4) Use encrypted messaging and email; (5) Consider VPN services. Emerging privacy developments: (1) New state privacy laws; (2) Federal privacy legislation proposals; (3) International data transfer restrictions; (4) AI and algorithmic accountability; (5) Biometric data protections. Consider that: (1) Privacy laws are state-specific; (2) Enforcement is still developing; (3) Corporate compliance varies; (4) Technology moves faster than law; (5) Personal vigilance remains important.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "privacy_law"
        },
        "extra_info": {
            "split": "train",
            "index": 21,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm thinking about filing for bankruptcy. What are the different types, how does the process work, and what are the long-term consequences?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Bankruptcy provides debt relief but has significant consequences requiring careful consideration. Main types: (1) Chapter 7 - liquidation of assets to pay creditors; (2) Chapter 13 - repayment plan over 3-5 years; (3) Chapter 11 - business reorganization; (4) Chapter 12 - family farmer/fisherman relief. Chapter 7 eligibility: (1) Must pass means test based on income; (2) Cannot have filed Chapter 7 in past 8 years; (3) Must complete credit counseling; (4) Assets may be liquidated (minus exemptions). Chapter 13 requirements: (1) Regular income source; (2) Secured debts under $1,395,875; (3) Unsecured debts under $465,275; (4) 3-5 year repayment plan; (5) Court approval required. Debts typically discharged: (1) Credit card debt; (2) Medical bills; (3) Personal loans; (4) Utility bills; (5) Some old tax debts. Non-dischargeable debts: (1) Student loans (except rare hardship); (2) Child support and alimony; (3) Recent tax debts; (4) Court fines and penalties; (5) Debts from fraud or willful injury. Bankruptcy process: (1) Credit counseling requirement; (2) Petition filing and documentation; (3) Automatic stay stops collection; (4) Meeting of creditors (341 meeting); (5) Discharge hearing. Long-term consequences: (1) Credit report impact for 7-10 years; (2) Difficulty obtaining credit initially; (3) Higher interest rates when credit available; (4) Employment and housing challenges; (5) Professional license impacts. Alternatives to consider: (1) Debt consolidation; (2) Debt settlement negotiations; (3) Credit counseling; (4) Informal payment plans; (5) Hardship programs. Before filing: (1) Gather financial documents; (2) Complete credit counseling; (3) Consider alternatives; (4) Consult bankruptcy attorney; (5) Understand local exemptions.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "bankruptcy_law"
        },
        "extra_info": {
            "split": "train",
            "index": 22,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm getting a divorce and want to know about dividing retirement accounts and pension benefits. What are the rules and tax implications?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Retirement account division in divorce involves complex rules and significant tax implications. Types of retirement accounts: (1) 401(k) and 403(b) plans; (2) Traditional and Roth IRAs; (3) Pension plans; (4) Government employee plans; (5) Military retirement benefits. Generally, retirement benefits earned during marriage are marital property subject to division. QDRO (Qualified Domestic Relations Order) requirements: (1) Required for 401(k), 403(b), and pension plans; (2) Must be approved by plan administrator; (3) Specifies payment amount and timing; (4) Protects against early withdrawal penalties; (5) Allows direct transfer to ex-spouse. IRA division process: (1) No QDRO required; (2) Transfer incident to divorce; (3) Must be direct trustee-to-trustee transfer; (4) Avoids taxes and penalties if done correctly; (5) Receiving spouse takes on tax obligations. Tax implications: (1) Transfers pursuant to divorce are generally tax-free; (2) Future withdrawals taxed to account owner; (3) Roth IRA transfers maintain tax-free status; (4) Early withdrawal penalties may apply; (5) Required minimum distributions continue. Valuation considerations: (1) Present value of future benefits; (2) Vesting schedules; (3) Survivor benefits; (4) Cost-of-living adjustments; (5) Early retirement options. Common mistakes: (1) Failing to get proper QDRO; (2) Incorrect tax withholding; (3) Not considering tax implications; (4) Ignoring survivor benefits; (5) Missing plan deadlines. Strategic considerations: (1) Timing of distributions; (2) Offsetting against other assets; (3) Survivor benefit elections; (4) Social Security implications; (5) Long-term financial planning. Consider consulting: (1) Divorce attorney familiar with retirement issues; (2) Financial planner; (3) Tax professional; (4) QDRO specialist; (5) Plan administrator.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 23,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I was injured in a slip and fall accident at a grocery store. What do I need to prove to have a successful premises liability claim?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Premises liability claims require proving the property owner's negligence caused your injury. Essential elements to prove: (1) Duty - owner owed you duty of care; (2) Breach - owner violated that duty; (3) Causation - breach caused your injury; (4) Damages - you suffered actual harm. Duty of care varies by visitor status: (1) Invitees (customers) - highest duty, reasonable care; (2) Licensees (social guests) - warn of known dangers; (3) Trespasser - very limited duty. For slip and fall cases, must prove: (1) Dangerous condition existed; (2) Condition caused the fall; (3) Owner knew or should have known about condition; (4) Owner failed to fix or warn about danger; (5) You were exercising reasonable care. Types of evidence to gather: (1) Incident report from store; (2) Photographs of scene and your injuries; (3) Witness statements; (4) Medical records and bills; (5) Surveillance video; (6) Store policies and procedures. Common dangerous conditions: (1) Wet floors without warning signs; (2) Uneven or damaged flooring; (3) Poor lighting; (4) Obstacles in walkways; (5) Inadequate maintenance. Defenses property owners raise: (1) Open and obvious danger; (2) Comparative negligence by plaintiff; (3) No actual or constructive notice; (4) Condition wasn't unreasonably dangerous; (5) Plaintiff caused own injury. Proving notice: (1) Actual notice - store knew about condition; (2) Constructive notice - condition existed long enough that store should have known; (3) Store created the condition; (4) Condition was recurring problem. Damages you can recover: (1) Medical expenses; (2) Lost wages; (3) Pain and suffering; (4) Future medical costs; (5) Permanent disability. Act quickly to: (1) Report incident immediately; (2) Seek medical attention; (3) Document everything; (4) Preserve evidence; (5) Consult personal injury attorney.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "personal_injury"
        },
        "extra_info": {
            "split": "train",
            "index": 24,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm dealing with a neighbor dispute over property boundaries. They built a fence that I believe is on my property. What are my legal options?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Property boundary disputes require careful documentation and may involve several legal remedies. First steps: (1) Locate your property deed and survey; (2) Review neighbor's property documents if available; (3) Hire licensed surveyor for accurate boundary determination; (4) Check local zoning and setback requirements; (5) Research property history and previous surveys. Legal theories available: (1) Trespass - fence encroaches on your property; (2) Nuisance - fence unreasonably interferes with property use; (3) Unjust enrichment - neighbor benefits from your land; (4) Quiet title action - establish clear ownership. Potential defenses neighbor might raise: (1) Adverse possession - open, notorious use for statutory period; (2) Boundary by acquiescence - long-standing boundary recognition; (3) Estoppel - you consented to or encouraged fence placement; (4) Prescriptive easement - right to use based on long use. Survey importance: (1) Professional survey establishes accurate boundaries; (2) Identifies encroachments and easements; (3) Provides evidence for legal proceedings; (4) May resolve dispute without litigation; (5) Required for title insurance claims. Resolution options: (1) Negotiate fence relocation; (2) Boundary line agreement; (3) Easement or license agreement; (4) Mediation or arbitration; (5) Litigation as last resort. Remedies if you prevail: (1) Injunction requiring fence removal; (2) Damages for trespass; (3) Quiet title establishing ownership; (4) Attorney fees (if state allows); (5) Restoration of property. Documentation needed: (1) Property deed and legal description; (2) Survey and plot plan; (3) Photographs of fence and property; (4) Communications with neighbor; (5) Evidence of property use and maintenance. Consider: (1) Cost of legal action vs. property value; (2) Relationship with neighbor; (3) Practical impact of encroachment; (4) Time and emotional investment; (5) Alternative dispute resolution first.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 25,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to adopt my stepchild whose biological father is absent. What's the legal process and what rights does the biological father have?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Stepparent adoption requires terminating the absent parent's rights and establishing new legal relationships. Legal requirements typically include: (1) Marriage to child's custodial parent; (2) Child's consent if over certain age (usually 12-14); (3) Termination of other parent's rights; (4) Home study (varies by state); (5) Court approval. Biological father's rights: (1) Constitutional due process protections; (2) Right to notice of adoption proceedings; (3) Right to contest termination; (4) Right to establish paternity if not already done; (5) Right to legal representation. Grounds for involuntary termination: (1) Abandonment for statutory period; (2) Failure to support child; (3) Failure to communicate with child; (4) Unfitness as parent; (5) Conviction of serious crimes. Abandonment typically requires: (1) No contact for specified time (often 6-12 months); (2) No financial support; (3) No communication or visitation attempts; (4) Intent to abandon (may be inferred). Process overview: (1) Petition for adoption filed; (2) Attempt to serve biological father; (3) Termination hearing if contested; (4) Home study if required; (5) Final adoption hearing. Notice requirements: (1) Personal service preferred; (2) Publication if father cannot be located; (3) Diligent search efforts required; (4) Proof of notice to court; (5) Appointed attorney if indigent. Factors courts consider: (1) Child's best interests; (2) Existing relationship with biological father; (3) Stability of stepparent relationship; (4) Child's wishes if age-appropriate; (5) Financial and emotional support. Benefits of adoption: (1) Full parental rights and responsibilities; (2) Inheritance rights; (3) Insurance and benefits eligibility; (4) Legal decision-making authority; (5) Emotional security for child. Consider: (1) Whether termination is truly in child's best interest; (2) Potential for future relationship with biological father; (3) Child's feelings about name change; (4) Impact on extended family relationships.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 26,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm starting an online business and need to know about sales tax obligations, privacy policies, and terms of service. What are the key legal requirements?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Online businesses face complex compliance requirements across multiple jurisdictions. Sales tax obligations post-Wayfair: (1) Economic nexus triggers vary by state; (2) Most states require registration once you exceed transaction/revenue thresholds; (3) Must collect tax from customers in nexus states; (4) Regular filing and remittance required; (5) Penalties for non-compliance. Privacy policy requirements: (1) California CCPA requires privacy policy for California residents; (2) EU GDPR if serving European customers; (3) COPPA if collecting children's data; (4) State privacy laws increasingly require disclosure; (5) Must be clear, accessible, and regularly updated. Essential privacy policy elements: (1) What data you collect; (2) How you use collected data; (3) Who you share data with; (4) User rights and choices; (5) Data security measures; (6) Contact information for privacy questions. Terms of service should address: (1) Acceptable use policies; (2) Liability limitations; (3) Dispute resolution procedures; (4) Intellectual property rights; (5) Termination conditions; (6) Governing law and jurisdiction. Additional compliance considerations: (1) Accessibility requirements (ADA); (2) Consumer protection laws; (3) Advertising and marketing regulations; (4) Industry-specific regulations; (5) International trade requirements if selling globally. E-commerce specific issues: (1) Return and refund policies; (2) Shipping and delivery terms; (3) Product liability considerations; (4) Payment processing compliance; (5) Subscription service regulations. Business structure considerations: (1) LLC or corporation for liability protection; (2) Business insurance including cyber liability; (3) Professional liability if providing services; (4) Contracts with vendors and suppliers; (5) Employee vs. contractor classifications. Implementation steps: (1) Consult attorney for compliance review; (2) Use privacy policy generators as starting point; (3) Implement data security measures; (4) Register for required business licenses; (5) Set up sales tax collection systems.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 27,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering a separation from my spouse before divorce. What are the legal implications of separation and how does it affect property, support, and custody?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Legal separation has significant implications and varies by state law. Types of separation: (1) Informal separation - living apart without court order; (2) Formal legal separation - court-ordered separation with terms; (3) Trial separation - temporary arrangement; (4) Permanent separation - indefinite arrangement. Legal separation effects: (1) Establishes separate living arrangements; (2) May trigger property division; (3) Can establish spousal support; (4) Addresses child custody and support; (5) Affects insurance and benefits. Property implications: (1) Assets acquired after separation may be separate property; (2) Debts incurred separately may be individual responsibility; (3) Joint accounts should be addressed; (4) Retirement benefits may stop accruing as marital property; (5) Real estate ownership and obligations continue. Spousal support considerations: (1) Temporary support during separation; (2) Factors include earning capacity and need; (3) Duration typically limited; (4) May be offset against future divorce settlement; (5) Tax implications for payer and recipient. Child custody and support: (1) Temporary custody arrangements; (2) Parenting time schedules; (3) Child support calculations; (4) Decision-making authority; (5) Modification procedures. Benefits of legal separation: (1) Provides structure and legal protection; (2) Addresses immediate financial needs; (3) Establishes custody arrangements; (4) Allows time to consider reconciliation; (5) May preserve certain benefits. Disadvantages: (1) Doesn't terminate marriage; (2) Can't remarry; (3) May delay final resolution; (4) Ongoing financial entanglements; (5) Additional legal costs. Separation vs. divorce: (1) Separation is potentially reversible; (2) Divorce terminates marriage permanently; (3) Some benefits preserved in separation; (4) Separation may be stepping stone to divorce; (5) Religious or personal reasons for separation. Important considerations: (1) Impact on taxes and benefits; (2) Insurance coverage continuation; (3) Social Security and retirement benefits; (4) Estate planning implications; (5) Effect on future divorce proceedings.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 28,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm worried about elder abuse of my grandmother by her caregiver. What constitutes elder abuse and what legal actions can I take to protect her?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Elder abuse is serious crime with both civil and criminal remedies available. Types of elder abuse: (1) Physical abuse - hitting, restraining, inappropriate medication; (2) Emotional abuse - threats, intimidation, isolation; (3) Financial abuse - theft, fraud, coercion; (4) Sexual abuse - unwanted sexual contact; (5) Neglect - failure to provide necessary care; (6) Self-neglect - inability to care for oneself. Warning signs to watch for: (1) Unexplained injuries or bruises; (2) Sudden changes in behavior or mood; (3) Missing personal belongings or money; (4) Poor hygiene or malnutrition; (5) Isolation from family and friends; (6) Caregiver controlling access to grandmother. Immediate steps: (1) Document evidence with photos and records; (2) Report to Adult Protective Services; (3) Contact local law enforcement if immediate danger; (4) Seek medical evaluation; (5) Consider emergency protective orders. Reporting requirements: (1) Most states mandate reporting suspected abuse; (2) Adult Protective Services investigates; (3) Medical professionals must report; (4) Anonymous reporting often allowed; (5) Immunity for good faith reporting. Legal remedies available: (1) Criminal prosecution of abuser; (2) Civil lawsuit for damages; (3) Restraining orders; (4) Guardianship or conservatorship; (5) Removal of caregiver. Civil litigation options: (1) Assault and battery claims; (2) Fraud and financial exploitation; (3) Negligence by care facilities; (4) Emotional distress damages; (5) Punitive damages for willful conduct. Guardianship considerations: (1) Court-appointed protection for incapacitated person; (2) Guardian makes decisions about care; (3) Financial conservatorship for money management; (4) Requires proof of incapacity; (5) Ongoing court supervision. Prevention strategies: (1) Background checks on caregivers; (2) Regular visits and communication; (3) Financial monitoring; (4) Clear care agreements; (5) Involving multiple family members. Resources available: (1) Elder law attorneys; (2) Adult Protective Services; (3) Elder abuse hotlines; (4) Local law enforcement; (5) Senior advocacy organizations.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "elder_law"
        },
        "extra_info": {
            "split": "train",
            "index": 29,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I received a speeding ticket and want to contest it. What are the options for fighting a traffic ticket and when should I hire an attorney?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Traffic ticket defense options depend on jurisdiction and specific circumstances. Common defense strategies: (1) Challenge officer's observations; (2) Question radar/laser equipment accuracy; (3) Argue necessity or emergency; (4) Dispute posted speed limits; (5) Procedural violations in citation. Options for contesting: (1) Plead not guilty and request trial; (2) Traffic school to avoid points; (3) Negotiate reduced charges; (4) Mitigation hearing to explain circumstances; (5) Deferred adjudication programs. Evidence to gather: (1) Photos of location and conditions; (2) Witness statements; (3) Equipment calibration records; (4) Weather and traffic conditions; (5) Officer's training and certification. When to hire attorney: (1) CDL license at risk; (2) Multiple violations or points; (3) Significant fines or consequences; (4) Possible license suspension; (5) Complex legal issues. Potential consequences: (1) Fines and court costs; (2) Points on driving record; (3) Insurance premium increases; (4) License suspension; (5) Employment impacts for drivers. Common defenses: (1) Radar detector wasn't properly calibrated; (2) Officer didn't have clear view; (3) Speedometer was inaccurate; (4) Emergency situation justified speed; (5) Mistaken identity. Traffic school benefits: (1) Avoid points on record; (2) Prevent insurance increases; (3) Dismiss ticket upon completion; (4) Learn safe driving practices; (5) Usually available for first offense. Negotiation strategies: (1) Request reduced charge; (2) Agree to traffic school; (3) Plead to non-moving violation; (4) Seek deferred adjudication; (5) Show driving record improvement. Court appearance tips: (1) Dress professionally; (2) Arrive early; (3) Bring all evidence; (4) Be respectful to judge; (5) Present clear, concise argument. Consider costs: (1) Attorney fees vs. fine amount; (2) Long-term insurance impacts; (3) Professional license consequences; (4) Time investment for trial; (5) Likelihood of success.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "traffic_law"
        },
        "extra_info": {
            "split": "train",
            "index": 30,
            "difficulty": "beginner",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering buying a franchise. What legal protections exist for franchisees and what should I look for in the franchise agreement?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Franchise investments require careful legal review due to significant financial commitments and ongoing obligations. Federal regulations: (1) FTC Franchise Rule requires disclosure document; (2) Franchise Disclosure Document (FDD) must be provided; (3) 14-day review period before signing; (4) Earnings claims must be substantiated; (5) Registration required in some states. Key FDD information: (1) Franchisor's business experience; (2) Bankruptcy and litigation history; (3) Initial fees and ongoing costs; (4) Territory rights and restrictions; (5) Franchisee obligations and restrictions. Essential contract terms to review: (1) Term length and renewal options; (2) Territorial exclusivity; (3) Fee structure and payment terms; (4) Training and support requirements; (5) Termination conditions and penalties. Red flags to avoid: (1) Pressure to sign quickly; (2) Guarantees of income or success; (3) Inadequate disclosure documents; (4) High franchisee turnover rates; (5) Ongoing litigation against franchisor. Financial considerations: (1) Initial franchise fee; (2) Royalty and advertising fees; (3) Equipment and inventory costs; (4) Working capital requirements; (5) Financing options and restrictions. Due diligence steps: (1) Speak with current and former franchisees; (2) Review franchisor's financial statements; (3) Analyze territory demographics; (4) Evaluate competition; (5) Assess training and support quality. Common franchisee rights: (1) Receive required disclosures; (2) Territory protection (if provided); (3) Training and support; (4) Use of trademarks and systems; (5) Renewal rights under certain conditions. Typical restrictions: (1) Operating procedures and standards; (2) Approved suppliers and products; (3) Advertising and marketing limitations; (4) Non-compete during and after term; (5) Approval for sale or transfer. Legal protections: (1) State franchise laws; (2) FTC enforcement actions; (3) Private right of action for violations; (4) Relationship laws in some states; (5) Contract law remedies. Before signing: (1) Hire franchise attorney; (2) Review all financial projections; (3) Understand all obligations; (4) Negotiate where possible; (5) Ensure adequate capitalization.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 31,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "My employer is requiring me to work mandatory overtime without compensation. What are my rights under wage and hour laws?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Wage and hour violations can result in significant back pay and penalties. Federal law basics: (1) Fair Labor Standards Act (FLSA) requires overtime pay; (2) Non-exempt employees must receive time-and-a-half for hours over 40; (3) Minimum wage requirements apply; (4) Recordkeeping requirements for employers; (5) Retaliation protection for complaints. Overtime requirements: (1) 1.5x regular rate for hours over 40 per week; (2) Daily overtime varies by state; (3) Consecutive workdays may trigger overtime; (4) On-call time may be compensable; (5) Travel time rules apply. Exempt vs. non-exempt status: (1) Executive, administrative, professional exemptions; (2) Computer professional exemption; (3) Outside sales exemption; (4) Must meet salary and duties tests; (5) Independent contractor misclassification. Common violations: (1) Failing to pay overtime; (2) Misclassifying employees as exempt; (3) Not paying for all hours worked; (4) Improper meal and rest break policies; (5) Requiring off-the-clock work. State law variations: (1) Higher minimum wages; (2) Daily overtime requirements; (3) Meal and rest break mandates; (4) Reporting pay requirements; (5) Expanded overtime eligibility. Documentation needed: (1) Time records and timesheets; (2) Pay stubs and payroll records; (3) Job descriptions and duties; (4) Company policies; (5) Communications about work hours. Remedies available: (1) Back pay for unpaid wages; (2) Liquidated damages (double damages); (3) Attorney fees and costs; (4) Reinstatement if terminated; (5) Injunctive relief. Filing process: (1) Department of Labor complaint; (2) State labor department claim; (3) Private lawsuit; (4) Class action if multiple employees; (5) Statute of limitations (2-3 years). Retaliation protection: (1) Cannot be fired for filing complaint; (2) Cannot be demoted or disciplined; (3) Protected activity includes cooperation with investigations; (4) Remedies for retaliation; (5) Whistleblower protections. Next steps: (1) Document all overtime worked; (2) Review employee handbook; (3) File complaint with appropriate agency; (4) Consult employment attorney; (5) Continue working while investigating.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 32,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm planning to write a book and want to understand copyright law. How do I protect my work and what can I do if someone infringes my copyright?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Copyright law provides automatic protection for original literary works with additional benefits from registration. Automatic copyright protection: (1) Exists from moment of creation; (2) Covers original works of authorship; (3) Protects expression, not ideas; (4) Lasts for author's life plus 70 years; (5) No registration required for basic protection. What copyright protects: (1) Original literary works; (2) Specific expression of ideas; (3) Plot, characters, dialogue; (4) Distinctive creative elements; (5) Derivative works based on original. What copyright doesn't protect: (1) Ideas, facts, concepts; (2) Titles and short phrases; (3) Common themes and plots; (4) Historical facts; (5) Works in public domain. Registration benefits: (1) Legal presumption of validity; (2) Required for infringement lawsuits; (3) Statutory damages up to $150,000; (4) Attorney fees in successful cases; (5) Customs protection against imports. Registration process: (1) Complete application with Copyright Office; (2) Pay registration fee ($45-$65); (3) Deposit copy of work; (4) Processing takes 3-6 months; (5) Certificate provides proof of registration. Fair use doctrine: (1) Criticism, comment, news reporting; (2) Teaching and scholarship; (3) Parody and satire; (4) Four-factor analysis required; (5) Case-by-case determination. Infringement remedies: (1) Injunctive relief to stop infringement; (2) Actual damages and profits; (3) Statutory damages if registered; (4) Attorney fees if registered; (5) Seizure of infringing copies. Publishing considerations: (1) Retain rights in publishing contract; (2) Understand grant of rights; (3) Reversion clauses; (4) Royalty and advance terms; (5) Termination rights. Digital protection: (1) DMCA takedown notices; (2) Digital rights management; (3) Online monitoring services; (4) Watermarking and metadata; (5) Terms of use for websites. International protection: (1) Berne Convention provides global protection; (2) Register in countries of publication; (3) Translation rights; (4) Moral rights in some countries; (5) Duration varies by country.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "intellectual_property"
        },
        "extra_info": {
            "split": "train",
            "index": 33,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm concerned about my elderly parent's capacity to make financial decisions. What legal options exist for protecting them from financial exploitation?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Protecting elderly parents from financial exploitation requires understanding capacity and available legal tools. Signs of diminished capacity: (1) Confusion about finances; (2) Unusual spending patterns; (3) Difficulty managing bills; (4) Susceptibility to scams; (5) Forgetting recent financial decisions. Legal capacity standards: (1) Ability to understand consequences; (2) Ability to communicate decisions; (3) Ability to reason about options; (4) Understanding of relevant information; (5) Consistent decision-making. Less restrictive options: (1) Financial power of attorney; (2) Joint bank accounts; (3) Automatic bill payment; (4) Trusted person oversight; (5) Representative payee for benefits. Power of attorney benefits: (1) Allows agent to act on parent's behalf; (2) Can be limited or broad; (3) Effective while parent has capacity; (4) Revocable by parent; (5) Less expensive than guardianship. Guardianship/conservatorship: (1) Court-appointed protection; (2) Required when capacity significantly impaired; (3) Guardian makes personal decisions; (4) Conservator manages finances; (5) Ongoing court supervision. Financial exploitation indicators: (1) Unexplained withdrawals; (2) New 'friends' with financial interest; (3) Changes to wills or beneficiaries; (4) Unpaid bills despite adequate funds; (5) Signed documents not understood. Protective measures: (1) Monitor bank accounts; (2) Limit access to large amounts; (3) Involve trusted professionals; (4) Report suspected abuse; (5) Consider guardianship if necessary. Reporting requirements: (1) Adult Protective Services; (2) Bank reporting of suspicious activity; (3) Elder abuse hotlines; (4) Law enforcement for criminal activity; (5) Professional licensing boards. Legal remedies: (1) Void unauthorized transactions; (2) Civil lawsuit for damages; (3) Criminal prosecution; (4) Injunctive relief; (5) Asset recovery. Planning strategies: (1) Create power of attorney while capable; (2) Establish revocable trust; (3) Regular financial reviews; (4) Involve multiple family members; (5) Professional money management. Court process: (1) Petition for guardianship; (2) Medical evaluation; (3) Court hearing; (4) Appointed attorney for parent; (5) Ongoing reporting requirements.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "elder_law"
        },
        "extra_info": {
            "split": "train",
            "index": 34,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm getting divorced and my spouse is hiding assets. What legal tools are available to find hidden assets and ensure fair property division?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Asset hiding in divorce requires aggressive discovery and forensic investigation. Discovery tools available: (1) Interrogatories requiring written answers; (2) Document requests for financial records; (3) Depositions under oath; (4) Subpoenas to third parties; (5) Court orders compelling disclosure. Financial documents to request: (1) Tax returns and business records; (2) Bank and investment account statements; (3) Credit card and loan documents; (4) Insurance policies and benefits; (5) Real estate and business valuations. Common hiding methods: (1) Transferring assets to family/friends; (2) Undervaluing business interests; (3) Overpaying taxes to create refunds; (4) Cryptocurrency and offshore accounts; (5) Delaying bonuses or income. Forensic accounting benefits: (1) Trace asset transfers; (2) Identify income sources; (3) Analyze spending patterns; (4) Uncover hidden accounts; (5) Value complex assets. Red flags to watch for: (1) Unusual financial transactions; (2) Secretive behavior about money; (3) Business expenses that seem personal; (4) Lifestyle inconsistent with reported income; (5) Reluctance to provide documentation. Court enforcement powers: (1) Contempt citations for non-compliance; (2) Sanctions including attorney fees; (3) Adverse inferences about hidden assets; (4) Freezing orders on accounts; (5) Appointment of neutral evaluators. Third-party discovery: (1) Banks and financial institutions; (2) Employers and business partners; (3) Accountants and financial advisors; (4) Insurance companies; (5) Government agencies. Remedies for hidden assets: (1) Unequal distribution favoring innocent spouse; (2) Monetary sanctions; (3) Contempt of court; (4) Set aside fraudulent transfers; (5) Extended spousal support. Documentation strategies: (1) Photograph and copy financial documents; (2) Monitor spending and account activity; (3) Preserve electronic records; (4) Identify all potential assets; (5) Track lifestyle and expenses. Professional assistance: (1) Divorce attorney with asset tracing experience; (2) Forensic accountant; (3) Private investigator; (4) Appraiser for valuable assets; (5) Vocational evaluator for earning capacity. Timing considerations: (1) Act quickly to preserve evidence; (2) Seek temporary restraining orders; (3) File discovery requests early; (4) Monitor ongoing financial activity; (5) Prepare for lengthy process.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 35,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I was wrongfully terminated from my job and believe it was discrimination. What evidence do I need and what's the process for filing a complaint?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Wrongful termination claims require proving discrimination based on protected characteristics. Protected classes include: (1) Race, color, national origin; (2) Gender and pregnancy; (3) Age (40 and older); (4) Disability; (5) Religion; (6) Sexual orientation (in some states). Evidence needed: (1) Documentation of discriminatory comments; (2) Comparative treatment of similarly situated employees; (3) Statistical evidence of disparate treatment; (4) Witness statements; (5) Performance reviews and disciplinary records. Types of discrimination: (1) Disparate treatment - intentional discrimination; (2) Disparate impact - neutral policy with discriminatory effect; (3) Retaliation for protected activity; (4) Harassment creating hostile environment; (5) Failure to accommodate disabilities. Filing process: (1) EEOC complaint within 180 days (300 in some states); (2) EEOC investigation; (3) Mediation opportunity; (4) Right to sue letter; (5) Federal court lawsuit within 90 days. Documentation to preserve: (1) Personnel file and performance reviews; (2) Communications with supervisors; (3) Witness contact information; (4) Company policies and procedures; (5) Evidence of continuing violations. Retaliation protection: (1) Filing EEOC complaint; (2) Participating in investigation; (3) Opposing discriminatory practices; (4) Requesting reasonable accommodations; (5) Reporting harassment or discrimination. Remedies available: (1) Reinstatement to position; (2) Back pay and benefits; (3) Front pay for future losses; (4) Compensatory damages; (5) Punitive damages; (6) Attorney fees. Wrongful termination defenses: (1) Legitimate business reason; (2) Poor performance; (3) Misconduct; (4) Reduction in force; (5) At-will employment. Strengthening your case: (1) Document everything immediately; (2) File complaint promptly; (3) Avoid signing separation agreements; (4) Preserve electronic communications; (5) Identify potential witnesses. Alternative dispute resolution: (1) Company internal grievance process; (2) Union grievance procedures; (3) EEOC mediation; (4) Private mediation; (5) Arbitration if required. Consider consulting employment attorney for: (1) Case evaluation; (2) EEOC filing assistance; (3) Investigation coordination; (4) Settlement negotiations; (5) Litigation representation. Time limits critical: (1) EEOC filing deadlines; (2) State agency requirements; (3) Preservation of evidence; (4) Witness availability; (5) Continuing violation doctrine.",
            "task_type": "general_chat",
            "jurisdiction": "federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 36,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to set up a trust for my children's education and future needs. What are the different types of trusts and their benefits?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Trusts provide flexible estate planning tools for transferring wealth and managing assets. Basic trust types: (1) Revocable living trust - can be changed or terminated; (2) Irrevocable trust - cannot be modified; (3) Testamentary trust - created through will; (4) Special needs trust - preserves government benefits; (5) Charitable trust - benefits charity. Education-focused trusts: (1) 529 education savings plans; (2) Educational trusts with specific purposes; (3) Custodial accounts under UTMA/UGMA; (4) Crummey trusts with education provisions; (5) Generation-skipping trusts. Trust benefits: (1) Avoid probate; (2) Provide asset protection; (3) Tax advantages; (4) Professional management; (5) Flexibility in distributions. Key trust parties: (1) Grantor - creates and funds trust; (2) Trustee - manages trust assets; (3) Beneficiary - receives distributions; (4) Successor trustee - backup management; (5) Trust protector - oversight role. Tax considerations: (1) Income tax on trust earnings; (2) Gift tax on contributions; (3) Generation-skipping transfer tax; (4) Estate tax implications; (5) Grantor trust rules. Distribution provisions: (1) Mandatory distributions at certain ages; (2) Discretionary distributions for needs; (3) Educational expense standards; (4) Health, education, maintenance, support; (5) Incentive provisions. Asset protection features: (1) Spendthrift provisions; (2) Discretionary distributions; (3) Independent trustees; (4) Domestic asset protection trusts; (5) Offshore trust structures. Trustee selection: (1) Family members; (2) Professional trustees; (3) Corporate trustees; (4) Co-trustees; (5) Directed trustees. Common mistakes: (1) Inadequate funding; (2) Poor trustee selection; (3) Inflexible terms; (4) Tax inefficiency; (5) Failure to update. Trust administration: (1) Investment management; (2) Tax filings; (3) Distribution decisions; (4) Record keeping; (5) Beneficiary communications. Costs involved: (1) Initial setup fees; (2) Ongoing trustee fees; (3) Tax preparation; (4) Legal and accounting costs; (5) Investment management fees. Consider professional help: (1) Estate planning attorney; (2) Tax professional; (3) Financial advisor; (4) Trust company; (5) Insurance specialist.",
            "task_type": "general_chat",
            "jurisdiction": "general",
            "legal_domain": "estate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 37,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm dealing with a homeowner's association that's being unreasonable about property modifications. What are my rights and how can I challenge HOA decisions?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "HOA disputes require understanding governing documents and state law protections. Governing documents hierarchy: (1) State HOA laws; (2) Declaration of covenants; (3) Bylaws; (4) Rules and regulations; (5) Architectural guidelines. Common HOA powers: (1) Enforce architectural standards; (2) Maintain common areas; (3) Collect assessments; (4) Adopt reasonable rules; (5) Impose fines for violations. Your rights as homeowner: (1) Review and inspect HOA records; (2) Due process for violations; (3) Attend and speak at meetings; (4) Vote on important matters; (5) Fair and uniform enforcement. Challenging HOA decisions: (1) Request hearing with board; (2) Present evidence and arguments; (3) Appeal to membership; (4) Seek mediation; (5) File lawsuit if necessary. Legal grounds for challenge: (1) Violation of governing documents; (2) Arbitrary or capricious enforcement; (3) Discriminatory treatment; (4) Procedural violations; (5) Exceeding authority. Common disputes: (1) Architectural review denials; (2) Selective enforcement; (3) Excessive fines; (4) Assessment disputes; (5) Access to records. Documentation needed: (1) All HOA communications; (2) Governing documents; (3) Similar approved projects; (4) Meeting minutes; (5) Enforcement patterns. Legal remedies: (1) Injunctive relief; (2) Monetary damages; (3) Attorney fees (if provided); (4) Declaratory judgment; (5) Mandamus to compel action. State law protections: (1) Fair enforcement requirements; (2) Due process mandates; (3) Record access rights; (4) Reasonable rule requirements; (5) Anti-discrimination provisions. Prevention strategies: (1) Read all governing documents; (2) Attend HOA meetings; (3) Get architectural approval before modifications; (4) Document all communications; (5) Build relationships with board members.",
            "task_type": "general_chat",
            "jurisdiction": "Colorado",
            "legal_domain": "property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 38,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm being harassed by debt collectors who call constantly and make threats. What are my rights under debt collection laws and how can I stop the harassment?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Debt collection harassment is regulated by federal and state laws with strong consumer protections. Fair Debt Collection Practices Act (FDCPA) prohibits: (1) Calling before 8 AM or after 9 PM; (2) Calling at work if told not to; (3) Threatening illegal action; (4) Using abusive or profane language; (5) Misrepresenting debt amount or legal status. Your rights include: (1) Demand written validation of debt; (2) Request collector stop calling; (3) Dispute inaccurate information; (4) Sue for FDCPA violations; (5) Report violations to authorities. Prohibited collection practices: (1) Threatening arrest or jail; (2) Claiming to be attorney if not; (3) Discussing debt with third parties; (4) Publishing your name; (5) Continuing contact after cease letter. Required disclosures: (1) Collector's identity; (2) Amount of debt; (3) Original creditor's name; (4) Right to dispute within 30 days; (5) Right to request validation. Stop harassment strategies: (1) Send written cease communication letter; (2) Request all communication in writing; (3) Record all phone calls (if legal); (4) Document all violations; (5) Change phone number if necessary. Debt validation process: (1) Request validation within 30 days; (2) Collector must provide proof; (3) Collection stops until validation provided; (4) Challenge incomplete validation; (5) Dispute inaccurate amounts. Legal remedies: (1) Actual damages; (2) Statutory damages up to $1,000; (3) Attorney fees and costs; (4) Injunctive relief; (5) Class action lawsuits. Common violations: (1) Robo-calling cell phones; (2) False threats of legal action; (3) Calling after cease letter; (4) Misrepresenting debt status; (5) Attempting to collect time-barred debt. Documentation needed: (1) All communications from collectors; (2) Phone records showing calls; (3) Proof of cease letter; (4) Validation requests; (5) Evidence of harassment. State law protections: (1) Additional prohibited practices; (2) Licensing requirements; (3) Shorter limitation periods; (4) Enhanced penalties; (5) Broader coverage. Filing complaints: (1) Consumer Financial Protection Bureau; (2) State attorney general; (3) Federal Trade Commission; (4) Better Business Bureau; (5) Professional licensing boards.",
            "task_type": "general_chat",
            "jurisdiction": "Federal",
            "legal_domain": "consumer_law"
        },
        "extra_info": {
            "split": "train",
            "index": 39,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to create a living will and healthcare directive. What should be included and how do I make sure it's legally valid?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Living wills and healthcare directives provide crucial guidance for end-of-life medical decisions. Essential components: (1) Life-sustaining treatment preferences; (2) Pain management instructions; (3) Organ donation decisions; (4) Healthcare proxy designation; (5) Specific medical scenarios. Legal requirements vary by state: (1) Written document; (2) Signed by competent adult; (3) Witnessed or notarized; (4) Specific formalities; (5) Revocation procedures. Life-sustaining treatments to address: (1) Artificial nutrition and hydration; (2) Mechanical ventilation; (3) Dialysis; (4) Antibiotics; (5) CPR and resuscitation. Healthcare proxy powers: (1) Make medical decisions when incapacitated; (2) Access medical records; (3) Communicate with doctors; (4) Authorize treatments; (5) Move between facilities. Important considerations: (1) Discuss wishes with family; (2) Inform doctors and healthcare providers; (3) Provide copies to proxy and family; (4) Review and update regularly; (5) Consider religious or cultural beliefs. Common scenarios to address: (1) Permanent vegetative state; (2) Terminal illness; (3) Dementia or cognitive decline; (4) Intensive care situations; (5) Emergency medical treatment. POLST (Physician Orders for Life-Sustaining Treatment): (1) Medical order form; (2) Signed by physician; (3) Specific treatment instructions; (4) Portable between facilities; (5) Complements living will. Making documents legally valid: (1) Follow state-specific requirements; (2) Use proper witnesses; (3) Avoid conflicts of interest; (4) Ensure mental capacity; (5) Consider attorney review. Distribution and storage: (1) Give copies to healthcare proxy; (2) Provide to primary care physician; (3) File with hospital; (4) Store in accessible location; (5) Carry wallet card. Regular updates needed: (1) After major illness; (2) Following family changes; (3) When moving states; (4) After new medical developments; (5) Periodically review preferences. Common mistakes: (1) Vague or ambiguous language; (2) Not discussing with family; (3) Failing to update; (4) Inadequate witnesses; (5) Contradictory instructions. Consider consulting: (1) Estate planning attorney; (2) Healthcare provider; (3) Religious advisor; (4) Family members; (5) Ethics committee if complex.",
            "task_type": "general_chat",
            "jurisdiction": "Pennsylvania",
            "legal_domain": "healthcare_law"
        },
        "extra_info": {
            "split": "train",
            "index": 40,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm starting a nonprofit organization and need to understand the legal requirements for 501(c)(3) status. What's the process and what ongoing obligations exist?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "501(c)(3) tax-exempt status requires meeting specific IRS requirements and ongoing compliance. Eligible purposes: (1) Charitable activities; (2) Religious purposes; (3) Educational activities; (4) Scientific research; (5) Prevention of cruelty to animals/children. Formation requirements: (1) Articles of incorporation; (2) Bylaws; (3) Board of directors; (4) Registered agent; (5) State nonprofit registration. IRS application process: (1) Form 1023 or 1023-EZ; (2) Filing fee ($275-$600); (3) Detailed activity description; (4) Financial projections; (5) Board member information. Required provisions: (1) Exempt purpose limitation; (2) Prohibition on private inurement; (3) Dissolution clause; (4) No political campaign activity; (5) Limited lobbying activities. Ongoing obligations: (1) Annual Form 990 filing; (2) State reporting requirements; (3) Board meetings and minutes; (4) Conflict of interest policies; (5) Document retention policies. Prohibited activities: (1) Political campaign intervention; (2) Excessive private benefit; (3) Prohibited lobbying amounts; (4) Unrelated business income; (5) Self-dealing transactions. Board governance: (1) Independent voting majority; (2) Conflict of interest policies; (3) Regular meetings; (4) Fiduciary responsibilities; (5) Executive compensation oversight. Financial management: (1) Separate bank accounts; (2) Accounting procedures; (3) Annual budgets; (4) Audit requirements; (5) Investment policies. Fundraising compliance: (1) State registration requirements; (2) Donor disclosure rules; (3) Charitable solicitation laws; (4) Grant application procedures; (5) Donor acknowledgment requirements. Employment law compliance: (1) Payroll taxes; (2) Workers' compensation; (3) Employment discrimination laws; (4) Wage and hour requirements; (5) Volunteer vs. employee classification. Common mistakes: (1) Inadequate record keeping; (2) Failure to file annual returns; (3) Excessive compensation; (4) Political activity; (5) Insufficient board oversight. Benefits of 501(c)(3) status: (1) Tax deduction for donors; (2) Federal income tax exemption; (3) Potential state tax exemptions; (4) Grant eligibility; (5) Postal rate discounts. Professional assistance: (1) Attorney for formation; (2) Accountant for financial management; (3) Consultant for grant writing; (4) Fundraising professionals; (5) Compliance specialists.",
            "task_type": "general_chat",
            "jurisdiction": "Federal",
            "legal_domain": "tax_law"
        },
        "extra_info": {
            "split": "train",
            "index": 41,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm involved in a car accident where the other driver was uninsured. What are my options for compensation and how does uninsured motorist coverage work?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Uninsured motorist accidents require understanding insurance coverage and alternative compensation sources. Uninsured motorist (UM) coverage: (1) Compensates for injuries from uninsured drivers; (2) May include underinsured coverage; (3) Required in many states; (4) Coverage limits vary; (5) Separate property damage coverage. Types of UM coverage: (1) Bodily injury coverage; (2) Property damage coverage; (3) Underinsured motorist coverage; (4) Hit-and-run coverage; (5) Stacked vs. non-stacked coverage. Filing UM claim process: (1) Report accident to your insurer; (2) Provide documentation; (3) Cooperate with investigation; (4) Submit medical records; (5) Negotiate settlement or arbitration. Recoverable damages: (1) Medical expenses; (2) Lost wages; (3) Pain and suffering; (4) Property damage; (5) Future medical costs. Alternative compensation sources: (1) At-fault driver's personal assets; (2) Employer liability if work-related; (3) Property owner liability; (4) Vehicle owner liability; (5) Dram shop liability. Pursuing uninsured driver: (1) Asset investigation; (2) Wage garnishment; (3) Property liens; (4) Future income streams; (5) Bankruptcy considerations. Common issues: (1) Coverage disputes; (2) Causation questions; (3) Comparative negligence; (4) Policy interpretation; (5) Settlement negotiations. Documentation needed: (1) Police report; (2) Medical records; (3) Witness statements; (4) Photos of damage; (5) Insurance policy information. State variations: (1) Mandatory vs. optional coverage; (2) Minimum coverage amounts; (3) Stacking rules; (4) Arbitration requirements; (5) Statute of limitations. Underinsured coverage: (1) Applies when at-fault driver's coverage insufficient; (2) Offset by other driver's payments; (3) Requires exhaustion of other coverage; (4) May require consent to settle; (5) Separate coverage limits. Hit-and-run coverage: (1) UM coverage applies; (2) Reporting requirements; (3) Investigation cooperation; (4) Physical contact requirements; (5) Phantom vehicle coverage. Legal strategies: (1) Maximize UM coverage; (2) Investigate all potential defendants; (3) Preserve evidence; (4) Avoid recorded statements; (5) Consider legal representation. Prevention: (1) Purchase adequate UM coverage; (2) Consider umbrella policies; (3) Understand policy terms; (4) Review coverage regularly; (5) Gap coverage for financed vehicles.",
            "task_type": "general_chat",
            "jurisdiction": "New_York",
            "legal_domain": "insurance_law"
        },
        "extra_info": {
            "split": "train",
            "index": 42,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm dealing with a contractor who did poor work on my home renovation. What legal remedies do I have and how can I recover my costs?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Contractor disputes involve breach of contract, warranty, and consumer protection claims. Common contractor problems: (1) Substandard workmanship; (2) Use of inferior materials; (3) Failure to complete work; (4) Cost overruns; (5) Code violations. Legal theories available: (1) Breach of contract; (2) Breach of implied warranty; (3) Negligence; (4) Fraud or misrepresentation; (5) Consumer protection violations. Contract requirements: (1) Written agreements preferred; (2) Detailed scope of work; (3) Materials specifications; (4) Timeline requirements; (5) Payment schedules. Evidence to gather: (1) Original contract and change orders; (2) Photos of defective work; (3) Expert evaluations; (4) Cost estimates for repairs; (5) Communications with contractor. Warranty protections: (1) Implied warranty of workmanship; (2) Materials warranties; (3) Building code compliance; (4) Fitness for particular purpose; (5) Merchantability warranties. Remedies available: (1) Completion of work; (2) Cost of repairs; (3) Diminished property value; (4) Consequential damages; (5) Punitive damages if fraud. State licensing requirements: (1) Contractor license verification; (2) Bonding requirements; (3) Insurance obligations; (4) Complaint procedures; (5) Disciplinary actions. Preliminary steps: (1) Document all problems; (2) Demand correction in writing; (3) Obtain repair estimates; (4) Stop payments if possible; (5) Preserve work area. Consumer protection laws: (1) Home improvement fraud statutes; (2) Cooling-off periods; (3) Lien law protections; (4) Bonding requirements; (5) Right to cancel contracts. Dispute resolution: (1) Direct negotiation; (2) Mediation services; (3) Arbitration if required; (4) Small claims court; (5) Civil litigation. Mechanic's lien issues: (1) Contractor's right to lien; (2) Subcontractor lien rights; (3) Lien deadlines; (4) Lien release procedures; (5) Lien waiver requirements. Recovery strategies: (1) Contractor's insurance claims; (2) Bonding company claims; (3) Licensing board complaints; (4) Credit card chargebacks; (5) Homeowner's insurance coverage. Prevention tips: (1) Verify licensing and insurance; (2) Check references thoroughly; (3) Get detailed written contracts; (4) Avoid large upfront payments; (5) Obtain lien waivers. Time limitations: (1) Statute of limitations; (2) Warranty periods; (3) Lien deadlines; (4) Insurance claim periods; (5) Licensing complaint deadlines.",
            "task_type": "general_chat",
            "jurisdiction": "Illinois",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 43,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I want to start a podcast and need to understand the legal issues around music licensing, guest interviews, and potential liability. What should I know?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Podcast legal issues span intellectual property, privacy, and liability concerns. Music licensing requirements: (1) Mechanical licenses for reproductions; (2) Synchronization licenses for audiovisual use; (3) Performance licenses from ASCAP/BMI/SESAC; (4) Master recording licenses; (5) Fair use considerations for brief excerpts. Guest interview considerations: (1) Obtain written consent to record; (2) Release forms for commercial use; (3) Privacy rights in conversations; (4) Defamation risks from guest statements; (5) Right of publicity issues. Content liability issues: (1) Defamation claims; (2) Privacy violations; (3) Copyright infringement; (4) Trademark issues; (5) Right of publicity violations. Business structure options: (1) Sole proprietorship; (2) LLC for liability protection; (3) Corporation for growth; (4) Partnership agreements; (5) Business insurance needs. Advertising and sponsorship: (1) FTC disclosure requirements; (2) Endorsement guidelines; (3) Contract terms with sponsors; (4) Tax implications; (5) Affiliate marketing rules. International considerations: (1) GDPR compliance for EU listeners; (2) Right to be forgotten; (3) International copyright laws; (4) Tax obligations; (5) Data transfer restrictions. Platform agreements: (1) Terms of service compliance; (2) Content policies; (3) Revenue sharing terms; (4) Termination provisions; (5) Intellectual property rights. Fair use for commentary: (1) Criticism and comment; (2) News reporting; (3) Educational purposes; (4) Transformative use; (5) Four-factor analysis. Privacy protection: (1) Anonymous sources; (2) Public figure exceptions; (3) Consent requirements; (4) Recording laws vary by state; (5) Confidentiality agreements. Trademark considerations: (1) Podcast name clearance; (2) Logo trademark; (3) Domain name issues; (4) Descriptive fair use; (5) Nominative fair use. Documentation needed: (1) Guest release forms; (2) Music licensing agreements; (3) Sponsor contracts; (4) Privacy policies; (5) Terms of use. Insurance options: (1) General liability; (2) Professional liability; (3) Media liability; (4) Cyber liability; (5) Errors and omissions. Legal disclaimers: (1) Entertainment purposes; (2) No professional advice; (3) Guest opinion disclaimers; (4) Accuracy limitations; (5) External link disclaimers.",
            "task_type": "general_chat",
            "jurisdiction": "Delaware",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 44,
            "difficulty": "intermediate",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering legal action against my insurance company for bad faith denial of my claim. What constitutes bad faith and what are my options?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Insurance bad faith claims require proving unreasonable conduct in claim handling. Bad faith elements: (1) Duty to act reasonably; (2) Breach of that duty; (3) Unreasonable conduct; (4) Causation of damages; (5) Actual damages suffered. Common bad faith practices: (1) Denying valid claims without investigation; (2) Unreasonable delays in processing; (3) Lowball settlement offers; (4) Misrepresenting policy terms; (5) Failure to defend insured. Types of bad faith: (1) First-party bad faith - your own insurer; (2) Third-party bad faith - other party's insurer; (3) Failure to settle within limits; (4) Failure to defend; (5) Breach of fiduciary duty. Investigation standards: (1) Prompt and thorough investigation; (2) Reasonable basis for denial; (3) Consideration of all evidence; (4) Expert opinions when needed; (5) Documentation of decision-making. Unreasonable delay factors: (1) Complexity of claim; (2) Insurer's resources; (3) Cooperation by insured; (4) Need for investigation; (5) Policy language clarity. Damages in bad faith cases: (1) Policy benefits owed; (2) Consequential damages; (3) Emotional distress; (4) Punitive damages; (5) Attorney fees. Evidence needed: (1) Insurance policy; (2) Claim file; (3) Communications with insurer; (4) Expert testimony; (5) Industry standards. Common defenses: (1) Reasonable basis for denial; (2) Policy exclusions apply; (3) Lack of coverage; (4) Insured's non-cooperation; (5) Legitimate coverage disputes. State law variations: (1) Unfair claims practices acts; (2) Statutory penalties; (3) Jury trial rights; (4) Damages limitations; (5) Attorney fee provisions. Before filing lawsuit: (1) Exhaust internal appeals; (2) File complaint with insurance commissioner; (3) Gather all documentation; (4) Obtain expert opinions; (5) Calculate actual damages. Proving bad faith: (1) Objective unreasonableness; (2) Pattern of misconduct; (3) Inadequate investigation; (4) Misrepresentation of facts; (5) Failure to follow procedures. Settlement considerations: (1) Strength of underlying claim; (2) Damages calculation; (3) Litigation costs; (4) Time factors; (5) Precedent value. Legal representation: (1) Insurance law experience; (2) Trial experience; (3) Contingency fee arrangements; (4) Resource requirements; (5) Track record with insurers. Time limitations: (1) Statute of limitations; (2) Policy deadlines; (3) Regulatory complaint periods; (4) Discovery rules; (5) Continuing violation doctrine.",
            "task_type": "general_chat",
            "jurisdiction": "New_York",
            "legal_domain": "insurance_law"
        },
        "extra_info": {
            "split": "train",
            "index": 45,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm going through a custody modification case and my ex-spouse wants to relocate with our children to another state. What factors do courts consider in relocation cases?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Child relocation cases involve complex analysis of competing interests and child welfare. Primary considerations: (1) Child's best interests; (2) Existing custody arrangement; (3) Reason for proposed move; (4) Impact on parent-child relationships; (5) Child's adjustment to new environment. Factors courts analyze: (1) Distance of proposed move; (2) Feasibility of preserving relationship with non-moving parent; (3) Child's age and maturity; (4) Quality of life improvements; (5) Extended family relationships. Legitimate reasons for relocation: (1) Employment opportunities; (2) Educational benefits; (3) Family support systems; (4) Financial necessities; (5) Remarriage considerations. Burden of proof standards: (1) Some states require 'clear and convincing evidence'; (2) Others use 'preponderance of evidence'; (3) Moving party typically bears burden; (4) Best interests standard applies; (5) State law variations significant. Evidence to present: (1) Detailed relocation plan; (2) Employment documentation; (3) Housing arrangements; (4) School district information; (5) Proposed visitation schedule. Objecting parent's arguments: (1) Disruption of existing relationships; (2) Inadequate visitation opportunities; (3) Financial hardship of travel; (4) Child's preference to stay; (5) Stability concerns. Court's protective measures: (1) Modified visitation schedules; (2) Extended summer visits; (3) Travel cost allocation; (4) Communication requirements; (5) Periodic review provisions. Technology considerations: (1) Video calls and virtual visitation; (2) Online school involvement; (3) Digital communication tools; (4) Scheduling platforms; (5) Long-distance parenting apps. Common restrictions: (1) Cannot move until court approval; (2) Advance notice requirements; (3) Temporary restraining orders; (4) Bond requirements; (5) Contempt penalties. Child's voice: (1) Guardian ad litem appointment; (2) In-camera interviews; (3) Custody evaluations; (4) Therapeutic assessments; (5) Age-appropriate consideration. Emergency situations: (1) Immediate safety concerns; (2) Temporary relocations; (3) Military deployments; (4) Domestic violence; (5) Protective orders. Modification procedures: (1) Petition for modification; (2) Service requirements; (3) Mediation attempts; (4) Custody evaluations; (5) Trial proceedings. International relocations: (1) Hague Convention considerations; (2) Passport restrictions; (3) International enforcement; (4) Treaty obligations; (5) Consular services. Prevention strategies: (1) Detailed custody orders; (2) Relocation clauses; (3) Geographic restrictions; (4) Consent requirements; (5) Penalty provisions.",
            "task_type": "general_chat",
            "jurisdiction": "Arizona",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 46,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm starting a tech company and need to understand employee vs. contractor classification. What are the legal tests and what are the risks of misclassification?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Worker classification affects taxes, benefits, and legal obligations significantly. IRS 20-factor test considerations: (1) Behavioral control - how work is performed; (2) Financial control - business aspects; (3) Relationship type - employee benefits, permanency; (4) Integration into business; (5) Investment in equipment. Key control factors: (1) Instructions on how to work; (2) Training provided; (3) Work schedule flexibility; (4) Location requirements; (5) Supervision level. Financial relationship factors: (1) Payment method - salary vs. project; (2) Expense reimbursement; (3) Investment in tools/equipment; (4) Profit/loss opportunity; (5) Multiple client relationships. Relationship indicators: (1) Written contract terms; (2) Employee benefits provided; (3) Permanency of relationship; (4) Work integral to business; (5) Exclusive services. Common misclassification risks: (1) Back taxes and penalties; (2) Unemployment insurance; (3) Workers' compensation; (4) Overtime violations; (5) Benefit entitlements. Federal agency tests: (1) IRS economic reality test; (2) Department of Labor test; (3) NLRB common law test; (4) EEOC control test; (5) State-specific variations. State law complications: (1) ABC test in many states; (2) Stricter contractor standards; (3) Presumption of employment; (4) Industry-specific rules; (5) Penalty enhancements. Technology sector specifics: (1) Specialized skills consideration; (2) Project-based work; (3) Remote work arrangements; (4) Intellectual property issues; (5) Stock option implications. Compliance strategies: (1) Proper contract drafting; (2) Consistent treatment; (3) Regular classification reviews; (4) Documentation practices; (5) Legal compliance audits. Safe harbor provisions: (1) Section 530 relief; (2) Consistent treatment; (3) Reasonable basis; (4) Reporting compliance; (5) No reclassification. Penalties for misclassification: (1) Employment taxes; (2) Interest and penalties; (3) Employee benefits; (4) Wage and hour violations; (5) State penalties. Best practices: (1) Written agreements; (2) Clear independence; (3) Multiple client relationships; (4) Separate business entity; (5) Professional liability insurance. Conversion considerations: (1) Voluntary compliance; (2) Amnesty programs; (3) Settlement opportunities; (4) Prospective corrections; (5) Audit preparedness. Industry trends: (1) Gig economy challenges; (2) Remote work implications; (3) Regulatory enforcement; (4) Legislative changes; (5) Court decisions.",
            "task_type": "general_chat",
            "jurisdiction": "Texas",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 47,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering a plea bargain in my criminal case. What should I know about plea negotiations and what rights am I giving up?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Plea bargaining involves waiving constitutional rights in exchange for concessions from prosecution. Types of plea agreements: (1) Charge bargaining - reduced charges; (2) Sentence bargaining - agreed sentence; (3) Fact bargaining - stipulated facts; (4) Count bargaining - dismissed counts; (5) Cooperation agreements. Rights waived in plea: (1) Right to trial by jury; (2) Right to confront witnesses; (3) Right against self-incrimination; (4) Right to require proof beyond reasonable doubt; (5) Right to appeal (usually limited). Plea requirements: (1) Voluntary and intelligent; (2) Factual basis for plea; (3) Understanding of consequences; (4) Competent counsel; (5) Judge's acceptance. Factors to consider: (1) Strength of prosecution's case; (2) Potential sentence if convicted; (3) Collateral consequences; (4) Personal circumstances; (5) Victim impact. Negotiation strategies: (1) Review discovery evidence; (2) Identify weaknesses in case; (3) Demonstrate mitigation factors; (4) Propose alternative sanctions; (5) Timing considerations. Collateral consequences: (1) Immigration implications; (2) Professional licensing; (3) Employment impacts; (4) Civil liability; (5) Future sentencing enhancements. Judge's role: (1) Ensure plea is voluntary; (2) Establish factual basis; (3) Explain consequences; (4) May reject agreement; (5) Sentencing authority. Withdrawal of plea: (1) Before sentencing - good cause; (2) After sentencing - limited circumstances; (3) Ineffective assistance; (4) Involuntary plea; (5) Breach by prosecution. Cooperation agreements: (1) Substantial assistance; (2) Truthful testimony; (3) Ongoing cooperation; (4) Sentence reduction; (5) Breach consequences. Protection from prosecution: (1) Immunity agreements; (2) Proffer agreements; (3) Queen for a day; (4) Use immunity; (5) Transactional immunity. Victim considerations: (1) Victim input rights; (2) Restitution requirements; (3) Protective orders; (4) Impact statements; (5) Notification requirements. Appeal limitations: (1) Waiver of appeal rights; (2) Exceptions to waiver; (3) Ineffective assistance claims; (4) Jurisdictional issues; (5) Sentence legality. Probation implications: (1) Probation conditions; (2) Supervision requirements; (3) Violation consequences; (4) Early termination; (5) Modification procedures. Before accepting plea: (1) Consult with attorney; (2) Understand all consequences; (3) Review evidence; (4) Consider alternatives; (5) Evaluate strength of case.",
            "task_type": "general_chat",
            "jurisdiction": "Ohio",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 48,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm dealing with a boundary dispute where my neighbor claims adverse possession of part of my property. What are the legal requirements for adverse possession and how can I protect my property?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Adverse possession allows someone to gain legal title through continuous occupation under specific conditions. Traditional requirements: (1) Actual possession - physical occupancy; (2) Open and notorious - visible use; (3) Exclusive possession - not shared with owner; (4) Hostile - without permission; (5) Continuous for statutory period. Statutory periods vary: (1) 5-30 years depending on state; (2) Payment of taxes may reduce period; (3) Color of title may affect timing; (4) Tacking by successive occupants; (5) Disability tolling provisions. Actual possession requirements: (1) Physical presence on property; (2) Use consistent with ownership; (3) Improvement or cultivation; (4) Enclosure or fencing; (5) Regular maintenance activities. Open and notorious standard: (1) Visible to reasonable inspection; (2) Use that alerts true owner; (3) Obvious to neighborhood; (4) Consistent with property type; (5) Notice to prudent owner. Hostility requirement: (1) Without owner's permission; (2) Claim of right; (3) Good faith belief unnecessary; (4) Objective test applied; (5) Permissive use defeats claim. Defenses against adverse possession: (1) Interrupt continuous possession; (2) Grant permission for use; (3) File lawsuit to eject; (4) Establish boundary markers; (5) Regular property inspections. Protective measures: (1) Survey property regularly; (2) Mark boundaries clearly; (3) Monitor neighbor activities; (4) Address encroachments immediately; (5) Document property use. Evidence in adverse possession: (1) Surveys and maps; (2) Photographs over time; (3) Witness testimony; (4) Tax payment records; (5) Improvement documentation. Exceptions and limitations: (1) Government property protected; (2) Registered land systems; (3) Minor or disabled owners; (4) Future interest holders; (5) Landlord-tenant relationships. Quiet title actions: (1) Establish clear ownership; (2) Resolve competing claims; (3) Marketable title; (4) Notice requirements; (5) Binding determinations. Color of title effect: (1) Defective deed; (2) Shorter limitation period; (3) Constructive possession; (4) Enhanced claims; (5) Good faith requirements. Boundary by acquiescence: (1) Alternative theory; (2) Mutual recognition; (3) Long-standing boundaries; (4) Practical locations; (5) Neighbor agreements. Modern limitations: (1) Recorded title systems; (2) Enhanced notice requirements; (3) Good faith standards; (4) Compensation requirements; (5) Policy restrictions. Prevention strategies: (1) Professional surveys; (2) Boundary agreements; (3) Licenses for use; (4) Regular inspections; (5) Prompt legal action.",
            "task_type": "general_chat",
            "jurisdiction": "Califonia",
            "legal_domain": "property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 49,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm considering joining a class action lawsuit as a plaintiff. What should I know about class actions, and what are the benefits and risks of participating?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Class action lawsuits allow groups of people with similar claims to sue together efficiently. Class action requirements: (1) Numerosity - too many for individual suits; (2) Commonality - shared legal/factual issues; (3) Typicality - representative claims typical of class; (4) Adequacy - fair representation of interests; (5) Superiority - better than individual suits. Types of class actions: (1) Damages class - monetary relief; (2) Injunctive relief - behavioral changes; (3) Declaratory judgment - legal determinations; (4) Settlement class - negotiated resolution; (5) Hybrid classes - multiple remedies. Benefits of participation: (1) Shared litigation costs; (2) Access to experienced attorneys; (3) Strength in numbers; (4) Efficient resolution; (5) Deterrent effect on defendants. Risks and limitations: (1) Limited control over litigation; (2) Potentially small individual recovery; (3) Binding judgment; (4) Attorney fee deductions; (5) Long time to resolution. Opt-out vs. opt-in: (1) Damages classes - opt-out typically; (2) Injunctive relief - automatic inclusion; (3) Notice requirements; (4) Deadline for opting out; (5) Individual suit preservation. Settlement considerations: (1) Fairness hearings; (2) Objection rights; (3) Attorney fee approval; (4) Distribution methods; (5) Claim procedures. Common class action areas: (1) Consumer fraud; (2) Securities violations; (3) Employment discrimination; (4) Product liability; (5) Data breach cases. Attorney compensation: (1) Contingency fees; (2) Court-approved fees; (3) Percentage of recovery; (4) Lodestar method; (5) Fee shifting provisions. Your rights as class member: (1) Receive notice of action; (2) Participate in settlement; (3) Object to settlement; (4) Opt out if permitted; (5) Monitor proceedings. Class certification process: (1) Motion to certify; (2) Discovery on class issues; (3) Hearing on certification; (4) Appeal rights; (5) Ongoing monitoring. Notice requirements: (1) Best notice practicable; (2) Individual notice if possible; (3) Publication notice; (4) Website notification; (5) Government agency notice. Evaluation factors: (1) Strength of individual claim; (2) Damages calculation; (3) Likelihood of success; (4) Alternative remedies; (5) Time and effort required. Red flags: (1) Inadequate representation; (2) Conflicts of interest; (3) Collusive settlements; (4) Excessive attorney fees; (5) Unfair distribution. Before joining: (1) Understand your individual claim; (2) Research class counsel; (3) Evaluate settlement terms; (4) Consider opting out; (5) Consult independent attorney if needed.",
            "task_type": "general_chat",
            "jurisdiction": "Georgia",
            "legal_domain": "civil_procedure"
        },
        "extra_info": {
            "split": "train",
            "index": 50,
            "difficulty": "advanced",
            "court_level": "n/a",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a motion to dismiss arguing that the plaintiff's breach of contract claim should be dismissed because the contract was illegal and unenforceable under California law."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION TO DISMISS: I. INTRODUCTION: Defendant moves to dismiss plaintiff's breach of contract claim under Rule 12(b)(6) as the alleged contract violates California public policy and is illegal and unenforceable. II. LEGAL STANDARD: A contract is void and unenforceable if its object is illegal or contravenes public policy. California courts will not enforce contracts that violate statutes or public policy. III. ARGUMENT: A. The Contract Violates California Labor Code: The alleged agreement requires [specific illegal activity] in direct violation of Labor Code Section [X]. California courts consistently refuse to enforce contracts requiring illegal performance. B. Public Policy Bars Enforcement: The contract's terms contravene California's strong public policy protecting [relevant protected interest]. In [relevant case], the California Supreme Court held that contracts violating fundamental public policies are void ab initio. C. Plaintiff Cannot Recover on Illegal Contract: Under the clean hands doctrine, plaintiff cannot seek judicial enforcement of an illegal agreement. California Civil Code Section 1598 prohibits recovery on illegal contracts. IV. CONCLUSION: The contract is illegal and unenforceable as a matter of law. Plaintiff's claim must be dismissed with prejudice. No amendment can cure the fundamental illegality. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "California",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 51,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["California Civil Code Section 1598"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's opposition to summary judgment in a personal injury case where the defendant claims the plaintiff was contributorily negligent."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S OPPOSITION TO SUMMARY JUDGMENT: I. INTRODUCTION: Defendant's motion for summary judgment must be denied because genuine issues of material fact exist regarding defendant's negligence and plaintiff's alleged contributory negligence. II. FACTUAL BACKGROUND: On [date], plaintiff was lawfully [activity] when defendant's [negligent conduct] caused [specific injury]. Evidence shows defendant violated multiple safety standards. III. LEGAL STANDARD: Summary judgment is inappropriate when reasonable minds could differ on material facts. All evidence must be viewed in the light most favorable to the non-moving party. IV. ARGUMENT: A. Genuine Issues Exist Regarding Defendant's Negligence: Defendant breached its duty of care by [specific failures]. Expert testimony establishes defendant's conduct fell below reasonable standards. Triable issues exist regarding causation and foreseeability. B. Comparative Fault is a Jury Question: Even if plaintiff bore some responsibility, comparative negligence is typically a jury question. The court cannot determine as a matter of law that plaintiff's conduct was the sole proximate cause. C. Defendant's Evidence is Insufficient: Defendant relies on self-serving testimony contradicted by physical evidence. Credibility determinations cannot be made on summary judgment. V. CONCLUSION: Material facts remain disputed. Defendant's motion should be denied and the case should proceed to trial. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 52,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief arguing that the plaintiff's Title VII discrimination claim should be dismissed because the plaintiff failed to exhaust administrative remedies."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S BRIEF: DISMISSAL FOR FAILURE TO EXHAUST ADMINISTRATIVE REMEDIES: I. STATEMENT OF ISSUE: Whether plaintiff's Title VII claim must be dismissed for failure to exhaust administrative remedies as required by 42 U.S.C. § 2000e-5(f)(1). II. SUMMARY OF ARGUMENT: Plaintiff's claim is barred because: (1) no timely EEOC charge was filed; (2) the judicial complaint raises claims not presented to the EEOC; (3) plaintiff failed to receive a right-to-sue letter. III. LEGAL STANDARD: Title VII requires exhaustion of administrative remedies as a jurisdictional prerequisite. Failure to exhaust mandates dismissal. IV. ARGUMENT: A. No Timely EEOC Charge Was Filed: Title VII requires EEOC charges within 180 days (or 300 days in deferral states). Plaintiff filed [X] days after the alleged discriminatory act, exceeding the statutory deadline. In [relevant case], the Supreme Court held untimely charges cannot confer jurisdiction. B. Judicial Claims Exceed Scope of EEOC Charge: Plaintiff's complaint alleges [specific claims] never presented to the EEOC. Courts lack jurisdiction over claims not reasonably related to the EEOC charge. C. No Right-to-Sue Letter: Plaintiff cannot demonstrate receipt of a proper right-to-sue letter as required by § 2000e-5(f)(1). V. CONCLUSION: Plaintiff's failure to exhaust administrative remedies bars this action. The complaint must be dismissed without prejudice. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 53,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["42 U.S.C. § 2000e-5(f)(1)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's opening statement in a wrongful termination case where the employee claims retaliation for whistleblowing."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S OPENING STATEMENT: Members of the jury, this case is about retaliation—pure and simple. When my client, [Name], discovered that [Company] was [illegal conduct], she did what any responsible employee would do: she reported it. And for doing the right thing, she was fired. The evidence will show that [Name] was a model employee for [X] years with outstanding performance reviews. In [date], she discovered [specific illegal conduct] and reported it to [authority]. Within [timeframe], she faced harassment, demotion, and ultimately termination. The timing is no coincidence. We will prove three things: First, [Name] engaged in protected activity by reporting [illegal conduct] in good faith. The law protects employees who report violations of [relevant statute]. Second, [Company] took adverse action by terminating her employment. There's no dispute she was fired. Third, there's a clear causal connection. The evidence shows [specific retaliatory acts] immediately following her whistleblowing. [Company] will try to claim she was fired for [pretextual reason]. But the evidence will show this is mere pretext. Her performance was exemplary until she blew the whistle. The law protects whistleblowers because society needs people willing to speak up about wrongdoing. [Company] tried to silence [Name], but you have the power to vindicate her rights and send a message that retaliation will not be tolerated. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 54,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's closing argument in a criminal case where the prosecution's case relies heavily on eyewitness identification."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S CLOSING ARGUMENT: Ladies and gentlemen of the jury, the prosecution wants you to convict [Defendant] based on mistaken eyewitness identification. But the evidence shows reasonable doubt permeates this case. The prosecution's case depends entirely on [witness's] identification. But consider the circumstances: It was [time of day], [weather conditions], from [distance], lasting only [duration]. These are precisely the conditions that lead to mistaken identification. The science is clear: eyewitness identification is unreliable. Studies show that [percentage] of wrongful convictions involve mistaken identification. Cross-racial identification, like we have here, is particularly problematic. [Witness] admitted uncertainty during [specific testimony]. The initial description of [height/weight/clothing] doesn't match my client. The lineup was [suggestive procedure]. Most telling, [witness] couldn't identify [Defendant] until [length of time] after the incident. Memory doesn't improve with time—it deteriorates. The prosecution has no physical evidence, no forensic evidence, no corroboration. They're asking you to convict based on one person's fallible memory under stressful conditions. That's not proof beyond a reasonable doubt. My client was [alibi evidence]. [Witness] confirming [defendant's] whereabouts at [time]. The prosecution must prove guilt beyond a reasonable doubt. They haven't met that burden. One person's uncertain identification under poor conditions cannot sustain a conviction. I ask you to find [Defendant] not guilty. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Florida",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 55,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief supporting a motion for preliminary injunction to stop a former employee from violating a non-compete agreement."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF IN SUPPORT OF MOTION FOR PRELIMINARY INJUNCTION: I. INTRODUCTION: Plaintiff seeks preliminary injunctive relief to prevent defendant from violating a valid non-compete agreement and causing irreparable harm to plaintiff's business. II. FACTUAL BACKGROUND: Defendant signed a non-compete agreement prohibiting [specific restrictions] for [duration] in [geographic area]. Defendant left plaintiff's employment and immediately joined competitor [Company], soliciting plaintiff's clients and using confidential information. III. LEGAL STANDARD: Preliminary injunctions require showing: (1) likelihood of success on merits; (2) irreparable harm; (3) balance of hardships favors plaintiff; (4) public interest supports relief. IV. ARGUMENT: A. Likelihood of Success: The non-compete agreement is valid and enforceable. It protects legitimate business interests including [customer relationships, confidential information, specialized training]. The restrictions are reasonable in scope, duration, and geographic area. B. Irreparable Harm: Plaintiff faces irreparable harm that cannot be compensated by monetary damages: (1) Loss of customer relationships built over [years]; (2) Disclosure of trade secrets and confidential information; (3) Damage to competitive position; (4) Diminished goodwill. C. Balance of Hardships: The hardship to plaintiff outweighs defendant's inconvenience. Defendant knew of the restrictions and chose to violate them. Plaintiff invested significantly in defendant's training and development. D. Public Interest: Enforcing valid contracts serves the public interest. The restrictions are reasonable and protect legitimate business interests. V. CONCLUSION: Plaintiff respectfully requests the Court grant preliminary injunctive relief. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "California",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 56,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's response to a motion to compel discovery in a products liability case where the plaintiff seeks internal safety testing documents."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S RESPONSE TO MOTION TO COMPEL DISCOVERY: I. INTRODUCTION: Defendant opposes plaintiff's motion to compel production of internal safety testing documents as overbroad, unduly burdensome, and seeking privileged information. II. FACTUAL BACKGROUND: Plaintiff seeks [specific documents] related to [product] testing conducted over [timeframe]. Defendant has already produced [volume] of documents including [categories]. III. LEGAL STANDARD: Discovery must be relevant, proportional, and not unduly burdensome. Privileged communications are protected from disclosure. IV. ARGUMENT: A. Request is Overbroad and Disproportionate: Plaintiff seeks documents spanning [timeframe] for [number] of products. This exceeds the scope of relevant discovery. The request is not proportional to the case's significance, complexity, or resources. B. Attorney-Client Privilege Protects Requested Documents: Many requested documents contain attorney-client communications regarding legal compliance. These communications are protected by attorney-client privilege and work product doctrine. C. Undue Burden and Expense: Producing requested documents would require [specific burden]. The cost exceeds the value of the claim. Alternative discovery methods could obtain relevant information more efficiently. D. Trade Secret Protection: Certain documents contain proprietary testing methods and formulations constituting trade secrets. Disclosure would harm defendant's competitive position. V. ALTERNATIVE PROPOSAL: Defendant proposes a protective order limiting disclosure of sensitive documents to counsel and designated experts. VI. CONCLUSION: Plaintiff's motion should be denied as overbroad and seeking privileged information. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Illinois",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 57,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for class certification in a consumer fraud case involving deceptive marketing practices."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF IN SUPPORT OF CLASS CERTIFICATION: I. INTRODUCTION: Plaintiff seeks certification of a class of consumers defrauded by defendant's deceptive marketing practices. The proposed class satisfies all requirements of Rule 23. II. PROPOSED CLASS DEFINITION: All persons who purchased [product] in [jurisdiction] between [dates] and were exposed to defendant's deceptive advertising claims. III. RULE 23(a) REQUIREMENTS: A. Numerosity: The class includes thousands of consumers based on defendant's sales data. Joinder is impracticable due to the number and geographic dispersion of class members. B. Commonality: Common questions include: (1) Whether defendant's advertising was deceptive; (2) Whether defendant violated consumer protection laws; (3) Whether class members suffered damages. C. Typicality: Representative plaintiff's claims are typical of the class. All members were exposed to the same deceptive marketing and suffered similar harm. D. Adequacy: Representative plaintiff will fairly protect class interests. Counsel has extensive experience in consumer class actions. IV. RULE 23(b)(3) REQUIREMENTS: A. Common Questions Predominate: The central issue—whether defendant's marketing was deceptive—is common to all class members. Individual reliance questions don't defeat commonality where advertising was uniform. B. Class Action is Superior: Individual actions are impracticable due to small damages. Class treatment promotes judicial efficiency and ensures wrongdoers face consequences. V. MANAGEABILITY: The case is manageable with standard notice procedures and damage calculations. VI. CONCLUSION: The proposed class satisfies all certification requirements. Plaintiff respectfully requests class certification. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "consumer_law"
        },
        "extra_info": {
            "split": "train",
            "index": 58,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Rule 23"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's motion for summary judgment in a copyright infringement case arguing fair use."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR SUMMARY JUDGMENT: FAIR USE DEFENSE: I. INTRODUCTION: Defendant moves for summary judgment because its use of plaintiff's copyrighted work constitutes fair use under 17 U.S.C. § 107. II. FACTUAL BACKGROUND: Defendant used portions of plaintiff's [work] in [context] for [purpose]. The use was transformative and served [educational/commentary/criticism] purposes. III. LEGAL STANDARD: Fair use is determined by four factors: (1) purpose and character of use; (2) nature of copyrighted work; (3) amount used; (4) effect on market. IV. ARGUMENT: A. Purpose and Character of Use (Factor 1): Defendant's use was transformative, creating new expression and meaning. The use was for [criticism/comment/parody/education], purposes explicitly protected by § 107. Unlike plaintiff's original work, defendant's use [specific transformation]. B. Nature of Copyrighted Work (Factor 2): Plaintiff's work is [factual/published], receiving thinner copyright protection. The work contains significant factual content not protected by copyright. C. Amount and Substantiality (Factor 3): Defendant used only [percentage/duration] necessary for its transformative purpose. The amount was reasonable for the intended [commentary/criticism]. No more was taken than necessary. D. Effect on Market (Factor 4): Defendant's use does not harm the market for plaintiff's work. The works serve different purposes and audiences. No evidence shows market harm or reduced licensing opportunities. V. CONCLUSION: All four factors favor fair use. Defendant's motion should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 59,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["17 U.S.C. § 107"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a prosecutor's closing argument in a white-collar fraud case involving securities violations."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PROSECUTOR'S CLOSING ARGUMENT: Ladies and gentlemen of the jury, this case is about greed, deception, and betrayal of trust. The defendant, [Name], systematically defrauded investors of millions of dollars through an elaborate securities fraud scheme. The evidence proves beyond a reasonable doubt that defendant committed securities fraud under 15 U.S.C. § 78j(b). We proved three elements: First, defendant made material misrepresentations. The testimony showed defendant told investors [specific lies about company performance/financial condition]. These weren't mere opinions—they were false statements of fact. Second, defendant acted with scienter—intent to deceive. The evidence shows defendant knew the statements were false. [Specific evidence of knowledge]. The documentary evidence proves defendant had access to the true financial information. Third, investors relied on these misrepresentations. [Victim testimony] showed they invested based on defendant's false representations. They lost their life savings because they trusted defendant. The defense claims these were mere business mistakes. But the evidence shows a pattern of deliberate deception spanning [timeframe]. The defendant created false documents, backdated contracts, and concealed losses. This wasn't negligence—it was criminal intent. The victims trusted defendant with their financial future. Defendant betrayed that trust for personal gain. The evidence demands a guilty verdict on all counts. Hold defendant accountable for this betrayal of trust. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 60,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["15 U.S.C. § 78j(b)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief opposing a motion for preliminary injunction in a trademark infringement case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S BRIEF IN OPPOSITION TO MOTION FOR PRELIMINARY INJUNCTION: I. INTRODUCTION: Plaintiff's motion for preliminary injunction should be denied because plaintiff cannot establish likelihood of success on the merits or irreparable harm. II. LEGAL STANDARD: Preliminary injunctions require: (1) likelihood of success on merits; (2) irreparable harm; (3) balance of hardships favors plaintiff; (4) public interest supports relief. All elements must be satisfied. III. ARGUMENT: A. Plaintiff Cannot Establish Likelihood of Success: 1. No Valid Trademark Rights: Plaintiff's mark is descriptive and lacks secondary meaning. The mark merely describes [characteristics] of the goods and cannot function as a trademark. 2. No Likelihood of Confusion: The eight-factor test defeats plaintiff's claim: (a) Different markets and customers; (b) Distinct appearance and packaging; (c) Different price points; (d) Sophisticated consumers; (e) No actual confusion evidence. B. No Irreparable Harm: Plaintiff cannot show irreparable harm absent evidence of actual consumer confusion or loss of control over reputation. Speculative harm is insufficient. Monetary damages can adequately compensate any harm. C. Balance of Hardships Favors Defendant: An injunction would force defendant to rebrand [product line] at enormous cost. Defendant invested heavily in marketing and distribution. The harm to defendant outweighs any harm to plaintiff. D. Public Interest Opposes Injunction: Competition benefits consumers. Restraining legitimate competition harms the public interest. IV. CONCLUSION: Plaintiff's motion should be denied. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 61,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for piercing the corporate veil in a breach of contract case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: PIERCING THE CORPORATE VEIL: I. INTRODUCTION: Plaintiff seeks to pierce the corporate veil and hold defendant [Individual] personally liable for [Corporation's] contractual obligations due to defendant's misuse of the corporate form. II. FACTUAL BACKGROUND: Defendant used [Corporation] as a mere instrumentality to avoid personal liability. The evidence shows: (1) Inadequate capitalization; (2) Commingling of assets; (3) Failure to observe corporate formalities; (4) Use of corporation to perpetrate fraud. III. LEGAL STANDARD: Corporate veil piercing requires showing: (1) Unity of interest between individual and corporation; (2) Inequitable result if corporate form is respected. IV. ARGUMENT: A. Unity of Interest Established: 1. Inadequate Capitalization: Corporation was formed with nominal capital insufficient for its business purpose. Defendant never adequately funded the corporation to meet its obligations. 2. Failure to Observe Formalities: No board meetings, corporate resolutions, or separate books. Defendant treated corporate assets as personal property. 3. Commingling of Assets: Defendant used corporate funds for personal expenses and vice versa. Bank accounts were used interchangeably. B. Inequitable Result: Respecting the corporate form would allow defendant to escape liability while retaining corporate benefits. Defendant used the corporation to defraud creditors and avoid personal responsibility. C. Fraud or Injustice: Defendant formed the corporation to avoid existing obligations and continue the same business without liability. This constitutes fraud on creditors. V. CONCLUSION: The corporate veil should be pierced to prevent injustice. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Delaware",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 62,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defense attorney's motion to suppress evidence in a DUI case where the stop was allegedly pretextual."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION TO SUPPRESS EVIDENCE: I. INTRODUCTION: Defendant moves to suppress all evidence obtained during an unconstitutional traffic stop lacking reasonable suspicion or probable cause. II. FACTUAL BACKGROUND: On [date], Officer [Name] stopped defendant's vehicle allegedly for [stated violation]. The stop led to DUI investigation and arrest. Body camera footage shows [specific facts]. III. LEGAL STANDARD: Fourth Amendment requires reasonable suspicion for investigatory stops. Evidence obtained in violation of constitutional rights must be suppressed. IV. ARGUMENT: A. No Reasonable Suspicion for Initial Stop: 1. Pretextual Stop: Officer's stated reason for the stop was pretextual. The evidence shows officer was conducting DUI patrol, not traffic enforcement. 2. No Traffic Violation: Video evidence shows defendant did not commit the alleged traffic violation. Defendant's driving was lawful and within traffic laws. B. Stop Exceeded Permissible Scope: 1. Prolonged Investigation: Officer extended the stop beyond its initial purpose without reasonable suspicion of additional criminal activity. 2. Improper Field Sobriety Tests: Tests were conducted without probable cause. Officer lacked reasonable suspicion that defendant was intoxicated. C. Tainted Evidence: All subsequent evidence (breath test, statements, observations) must be suppressed as fruit of the poisonous tree. The illegal stop taints all evidence obtained thereafter. D. No Inevitable Discovery: The State cannot establish that evidence would have been discovered through lawful means. V. CONCLUSION: The stop violated defendant's Fourth Amendment rights. All evidence must be suppressed. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Texas",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 63,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief supporting a motion for attorney fees in a successful civil rights case under 42 U.S.C. § 1988."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF IN SUPPORT OF ATTORNEY FEES: I. INTRODUCTION: Plaintiff seeks attorney fees under 42 U.S.C. § 1988 as the prevailing party in this civil rights action. The requested fees are reasonable and necessary. II. LEGAL STANDARD: Section 1988 provides that courts may award reasonable attorney fees to prevailing parties in civil rights cases. Fees are calculated using the lodestar method: reasonable hours × reasonable hourly rate. III. ARGUMENT: A. Plaintiff is the Prevailing Party: Plaintiff obtained substantial relief through [judgment/settlement]. The relief achieved material alteration of the legal relationship between parties and provided significant benefit to plaintiff. B. Reasonable Hourly Rates: The requested rates of $[amount] for partners and $[amount] for associates are reasonable for this jurisdiction. The rates are supported by: (1) Affidavits of counsel; (2) Market rate surveys; (3) Rates awarded in similar cases. C. Reasonable Hours Expended: Counsel spent [number] hours on this case. The time records show: (1) Detailed contemporaneous time entries; (2) Work directly related to civil rights claims; (3) Necessary and productive activities. D. Complexity and Results Achieved: This case involved complex constitutional issues requiring specialized expertise. The excellent results achieved justify the time spent. E. No Reduction Warranted: Plaintiff's success was substantial and the hours were reasonably expended. No reduction in the lodestar amount is appropriate. IV. CONCLUSION: Plaintiff respectfully requests attorney fees in the amount of $[total]. The fees are reasonable and necessary. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "civil_rights_law"
        },
        "extra_info": {
            "split": "train",
            "index": 64,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["42 U.S.C. § 1988"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's opening statement in a medical malpractice case involving failure to diagnose cancer."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S OPENING STATEMENT: Members of the jury, this case is about medicine, not magic. The plaintiff wants you to believe that Dr. [Name] should have diagnosed cancer during a routine examination. But the evidence will show that Dr. [Name] provided excellent care that met all medical standards. Cancer is a devastating disease, and we sympathize with the plaintiff's suffering. But tragedy doesn't equal malpractice. The evidence will show that Dr. [Name] acted reasonably at every step. When plaintiff first presented with [symptoms], Dr. [Name] conducted a thorough examination. The symptoms were consistent with [common conditions], not cancer. Dr. [Name] ordered appropriate tests including [specific tests]. The results were [normal/within normal limits]. Dr. [Name] provided appropriate follow-up care and instructions. The plaintiff's expert will claim Dr. [Name] should have [specific actions]. But our experts will testify that Dr. [Name]'s care met the standard of care. A reasonable physician in the same circumstances would have acted similarly. The plaintiff's cancer was [type] which is [characteristics]. This type of cancer is difficult to diagnose in early stages. The medical literature shows that [statistical evidence]. Even with perfect care, some cancers are not detected until later stages. The law requires that we prove Dr. [Name] acted reasonably under the circumstances. The evidence will show excellent medical care. Medicine has limitations, and physicians cannot be expected to diagnose every condition immediately. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Pennsylvania",
            "legal_domain": "medical_malpractice_law"
        },
        "extra_info": {
            "split": "train",
            "index": 65,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for damages in a breach of contract case involving lost profits."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: DAMAGES FOR LOST PROFITS: I. INTRODUCTION: Plaintiff seeks lost profits damages resulting from defendant's material breach of contract. The lost profits are foreseeable, certain, and properly calculated. II. LEGAL STANDARD: Lost profits are recoverable when: (1) reasonably certain; (2) foreseeable at contract formation; (3) caused by defendant's breach; (4) properly calculated. III. ARGUMENT: A. Lost Profits are Reasonably Certain: 1. Established Business: Plaintiff operated [business] for [years] with consistent profitability. Historical financial data establishes reliable profit patterns. 2. Specific Contract Performance: The breached contract involved [specific performance] that would have generated [specific profits]. The profits are calculable based on contract terms and established business operations. B. Foreseeability: At contract formation, defendant knew or should have known that breach would cause lost profits. The parties discussed [specific representations] about anticipated profits. Defendant understood plaintiff's business model and profit expectations. C. Causation: Defendant's breach directly caused the lost profits. But for defendant's breach, plaintiff would have earned the projected profits. No intervening factors prevent recovery. D. Proper Calculation: Expert testimony establishes lost profits of $[amount] based on: (1) Historical profit margins; (2) Market conditions; (3) Contract specifications; (4) Proven methodology. E. Mitigation: Plaintiff reasonably attempted to mitigate damages by [specific actions]. The lost profits reflect damages after mitigation. IV. CONCLUSION: Plaintiff is entitled to lost profits damages of $[amount]. The damages are certain, foreseeable, and proven. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 66,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's motion for judgment as a matter of law in a product liability case after plaintiff's case-in-chief."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR JUDGMENT AS A MATTER OF LAW: I. INTRODUCTION: Defendant moves for judgment as a matter of law under Rule 50(a) because plaintiff failed to establish essential elements of product liability claims. II. LEGAL STANDARD: Judgment as a matter of law is appropriate when the evidence is insufficient for a reasonable jury to find for the non-moving party. All evidence must be viewed in the light most favorable to plaintiff. III. ARGUMENT: A. No Evidence of Product Defect: 1. Design Defect: Plaintiff failed to prove the product's design was unreasonably dangerous. No evidence shows alternative design that would have prevented injury. Expert testimony was speculative and unsupported. 2. Manufacturing Defect: No evidence shows the product deviated from design specifications. Product examination revealed no manufacturing flaws. B. No Evidence of Causation: 1. Factual Causation: Plaintiff failed to prove the alleged defect caused the injury. The evidence shows [alternative causes] could have caused the accident. 2. Proximate Causation: The injury was not a foreseeable result of any alleged defect. Intervening causes broke the causal chain. C. Assumption of Risk: Plaintiff assumed known risks by [specific conduct]. The dangers were open and obvious. Plaintiff's conduct was unreasonable. D. Statute of Limitations: Plaintiff's claims are barred by the statute of limitations. The injury occurred more than [years] before filing suit. IV. CONCLUSION: Plaintiff failed to establish essential elements of product liability claims. Defendant's motion should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "California",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 67,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["Rule 50(a)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for injunctive relief in an environmental law case involving water pollution."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR INJUNCTIVE RELIEF: I. INTRODUCTION: Plaintiff seeks permanent injunctive relief to stop defendant's ongoing violation of environmental laws and prevent irreparable harm to [water body/ecosystem]. II. FACTUAL BACKGROUND: Defendant's [facility] has discharged pollutants into [water body] in violation of the Clean Water Act. The discharge contains [specific pollutants] at levels exceeding permitted limits. III. LEGAL STANDARD: Injunctive relief requires: (1) likelihood of success on merits; (2) irreparable harm; (3) balance of hardships favors plaintiff; (4) public interest supports relief. IV. ARGUMENT: A. Likelihood of Success on Merits: 1. Clean Water Act Violation: Defendant violated 33 U.S.C. § 1311 by discharging pollutants without a permit or in excess of permit limits. Monitoring data shows [specific violations]. 2. Continuing Violation: The discharge is ongoing and continuous. Defendant has failed to take corrective action despite notice. B. Irreparable Harm: 1. Environmental Damage: Pollution causes irreparable harm to aquatic ecosystems that cannot be adequately compensated by monetary damages. 2. Public Health Risk: Contaminated water poses health risks to [community/users]. The harm is immediate and continuing. C. Balance of Hardships: Environmental protection outweighs economic hardship to defendant. Defendant can comply through available pollution control technology. The cost of compliance is reasonable compared to environmental harm. D. Public Interest: The public interest strongly favors environmental protection. Clean water is essential for public health and ecological integrity. V. CONCLUSION: Plaintiff respectfully requests permanent injunctive relief requiring defendant to cease polluting discharges. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "environmental_law"
        },
        "extra_info": {
            "split": "train",
            "index": 68,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["33 U.S.C. § 1311"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief arguing for dismissal of a § 1983 civil rights claim based on qualified immunity."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S BRIEF: QUALIFIED IMMUNITY DEFENSE: I. INTRODUCTION: Defendant Officer [Name] is entitled to qualified immunity from plaintiff's § 1983 claims because defendant's actions did not violate clearly established law. II. LEGAL STANDARD: Qualified immunity protects government officials from liability unless: (1) constitutional right was violated; (2) right was clearly established at the time of the conduct. III. ARGUMENT: A. No Constitutional Violation: 1. Reasonable Force: The force used was objectively reasonable under the circumstances. Defendant faced [specific threat/resistance]. The Graham factors support defendant's actions. 2. Probable Cause: Defendant had probable cause to arrest plaintiff based on [specific facts]. The arrest was lawful and justified. B. Right Was Not Clearly Established: 1. No Controlling Precedent: No Supreme Court, Circuit, or state supreme court decision established that defendant's specific conduct violated constitutional rights. 2. Factual Distinction: Existing cases are factually distinguishable. The specific circumstances here were not addressed by prior precedent. 3. Reasonable Officer Standard: A reasonable officer in defendant's position would not have known the conduct was unlawful. The law was not clearly established with sufficient specificity. C. Objective Reasonableness: Even if a violation occurred, defendant's actions were objectively reasonable. Defendant acted in good faith based on training and experience. The situation required split-second decisions under dangerous circumstances. IV. CONCLUSION: Defendant is entitled to qualified immunity. Plaintiff's § 1983 claims should be dismissed. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "civil_rights_law"
        },
        "extra_info": {
            "split": "train",
            "index": 69,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["§ 1983"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's closing argument in a personal injury case involving a slip and fall at a retail store."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S CLOSING ARGUMENT: Members of the jury, this case is about accountability and responsibility. [Store] failed in its duty to keep its premises safe, and my client [Name] suffered serious injuries as a result. The evidence shows that on [date], [Name] was shopping at [Store] when she slipped on [substance] in aisle [number]. The substance had been on the floor for [duration] before the accident. [Store] knew or should have known about this dangerous condition. The surveillance video shows [Store] employees walked past the spill multiple times without cleaning it. [Store] had a duty to maintain safe premises for customers. They failed. The evidence shows [Store] violated its own safety policies. Their employee manual requires [specific procedures] for spill cleanup. These procedures weren't followed. [Store] will claim they didn't know about the spill. But ignorance is not a defense when it results from failure to inspect. The evidence shows [Store] failed to conduct reasonable inspections. [Name] suffered [specific injuries] requiring [medical treatment]. The medical evidence shows [permanent limitations]. She was [age] and active before this accident. Now she struggles with [specific limitations]. The economic damages are [amount] for medical expenses and lost wages. But the pain and suffering cannot be measured in dollars. [Name] will live with these injuries forever. [Store] chose profits over safety. They could have prevented this accident with reasonable care. Your verdict should hold them accountable and provide fair compensation for [Name]'s injuries. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Florida",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 70,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief opposing a motion for class certification in an employment discrimination case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S BRIEF OPPOSING CLASS CERTIFICATION: I. INTRODUCTION: Plaintiff's motion for class certification should be denied because the proposed class fails to satisfy the requirements of Rule 23(a) and (b)(3). II. FACTUAL BACKGROUND: Plaintiff seeks to represent a class of [number] employees claiming [type] discrimination. The proposed class spans [timeframe] and includes employees in [different positions/locations]. III. ARGUMENT: A. Rule 23(a) Requirements Not Met: 1. Lack of Commonality: Plaintiff cannot identify common questions of law or fact. Each employee's experience is individualized based on: (a) Different supervisors and decision-makers; (b) Varying performance evaluations; (c) Different job functions and qualifications; (d) Individual circumstances affecting employment decisions. 2. Typicality Absent: Representative plaintiff's claims are not typical of the class. Plaintiff's [specific circumstances] are unique and not representative of other employees' experiences. 3. Inadequate Representation: Representative plaintiff cannot fairly represent the class due to [conflicts of interest/inadequate knowledge]. B. Rule 23(b)(3) Requirements Not Satisfied: 1. Individual Issues Predominate: Each claim requires individualized proof of: (a) Discriminatory intent toward specific employee; (b) Causation between protected characteristic and adverse action; (c) Individual damages and harm. 2. Class Action Not Superior: Individual actions are superior because: (a) Manageable number of potential plaintiffs; (b) Individual claims are substantial; (c) Existing individual litigation. C. Manageability Problems: The proposed class is unmanageable due to [specific issues]. Different statutes of limitations and legal standards apply to different class members. IV. CONCLUSION: Class certification should be denied. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 71,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Rule 23(a)", "Rule 23(b)(3)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a prosecutor's opening statement in a domestic violence case involving repeated abuse."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PROSECUTOR'S OPENING STATEMENT: Members of the jury, this case is about power, control, and escalating violence. The evidence will show that the defendant systematically abused [victim] over [timeframe], turning what should have been a safe relationship into a nightmare of fear and violence. The defendant stands charged with [specific charges]. The evidence will prove each element beyond a reasonable doubt. On [date], the defendant [specific violent act]. But this wasn't an isolated incident. The evidence will show a pattern of abuse that escalated over time. [Victim] will testify about the defendant's controlling behavior: monitoring her activities, isolating her from friends and family, and using threats and intimidation. You'll hear about previous incidents: [specific examples]. Each time, the violence became more severe. The defendant's behavior follows a classic cycle of domestic violence: tension building, violent explosion, then false promises to change. The evidence includes: [medical records, photographs, witness testimony, 911 calls]. You'll see [victim's] injuries and hear her terrified voice calling for help. The defendant will likely claim this was mutual combat or self-defense. But the evidence shows defendant was the primary aggressor. The injuries, witness testimony, and physical evidence contradict any claim of self-defense. Domestic violence thrives in silence and isolation. [Victim] found the courage to speak out and seek help. The law provides protection for victims, but it requires your participation. The evidence demands accountability. Hold the defendant responsible for these crimes. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Illinois",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 72,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for specific performance in a real estate contract dispute."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR SPECIFIC PERFORMANCE: I. INTRODUCTION: Plaintiff seeks specific performance of a real estate purchase agreement because monetary damages are inadequate and the property is unique. II. FACTUAL BACKGROUND: Plaintiff and defendant entered into a purchase agreement for [property description] on [date] for $[amount]. Plaintiff performed all conditions but defendant refuses to convey title. III. LEGAL STANDARD: Specific performance is available when: (1) valid contract exists; (2) plaintiff performed; (3) defendant breached; (4) monetary damages are inadequate; (5) contract terms are certain. IV. ARGUMENT: A. Valid Enforceable Contract: The parties executed a written purchase agreement containing all essential terms: parties, property description, price, and closing date. The contract satisfies the statute of frauds. B. Plaintiff's Performance: Plaintiff performed all contractual obligations: (1) Paid earnest money; (2) Obtained financing pre-approval; (3) Completed inspection within deadline; (4) Satisfied all contingencies. C. Defendant's Breach: Defendant materially breached by refusing to convey title at closing. Defendant's stated reasons [specific reasons] are pretextual and unsupported. D. Inadequate Legal Remedy: Monetary damages cannot adequately compensate plaintiff because: (1) Real property is unique; (2) Specific location and characteristics cannot be replicated; (3) Market conditions may prevent finding substitute property; (4) Plaintiff has emotional attachment to the property. E. Definite and Certain Terms: Contract terms are sufficiently definite for court enforcement. All essential terms are clearly stated. F. Clean Hands: Plaintiff acted in good faith throughout the transaction. No misconduct or inequitable conduct occurred. V. CONCLUSION: Plaintiff respectfully requests specific performance requiring defendant to convey title. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Colorado",
            "legal_domain": "real_estate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 73,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defense attorney's closing argument in a burglary case where the evidence is largely circumstantial."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENSE CLOSING ARGUMENT: Ladies and gentlemen of the jury, the prosecution wants you to convict my client based on speculation, assumptions, and circumstantial evidence. But the law requires proof beyond a reasonable doubt—not guesswork. The prosecution claims my client burglarized [location] on [date]. But where's the evidence? There are no fingerprints, no DNA, no witnesses who saw my client at the scene. The prosecution's case is built on: [specific circumstantial evidence]. But each piece has an innocent explanation. The fact that my client was [location/activity] doesn't prove burglary. The prosecution wants you to connect dots that shouldn't be connected. Consider what's missing: No physical evidence linking my client to the crime scene. No witness identification. No stolen property found in my client's possession. The prosecution's timeline doesn't work. [Specific timeline problems]. My client couldn't have been at the crime scene when they claim. The prosecution's theory requires you to speculate about my client's intentions and actions. But speculation is not evidence. The burden is on the prosecution to prove guilt beyond a reasonable doubt. They haven't met that burden. Circumstantial evidence can support a conviction, but only when it points to guilt and is inconsistent with any reasonable hypothesis of innocence. Here, the evidence is consistent with my client's innocence. The prosecution has failed to eliminate reasonable doubt. My client is presumed innocent, and that presumption hasn't been overcome. I ask you to find my client not guilty. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Ohio",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 74,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for punitive damages in a fraud case against a financial advisor."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR PUNITIVE DAMAGES: I. INTRODUCTION: Plaintiff seeks punitive damages against defendant financial advisor for deliberate fraud that betrayed client trust and violated professional duties. II. LEGAL STANDARD: Punitive damages require proof by clear and convincing evidence of: (1) actual malice; (2) deliberate fraud; (3) oppression; or (4) conduct showing willful and wanton disregard for others' rights. III. FACTUAL BACKGROUND: Defendant, a licensed financial advisor, systematically defrauded elderly clients by [specific fraudulent scheme]. Defendant stole over $[amount] from [number] clients over [timeframe]. IV. ARGUMENT: A. Clear and Convincing Evidence of Fraud: 1. Deliberate Deception: Defendant knowingly made false representations about [investment opportunities/account performance]. Documents prove defendant created fake statements and forged signatures. 2. Intent to Defraud: Defendant's conduct was calculated to deceive clients. The elaborate scheme shows premeditation and intent to steal client funds. B. Oppression and Malice: 1. Targeting Vulnerable Victims: Defendant specifically targeted elderly clients with diminished capacity. This shows oppression and malice toward vulnerable populations. 2. Breach of Fiduciary Duty: Defendant violated professional duties and exploited position of trust. The betrayal of fiduciary duty supports punitive damages. C. Deterrent Effect Necessary: 1. Professional Deterrence: Punitive damages are necessary to deter similar conduct by financial professionals. The financial industry must be held accountable for fraud. 2. Compensation for Harm: Defendant's conduct caused not just financial harm but emotional distress and loss of trust. D. Defendant's Financial Capacity: Defendant has substantial assets and income capable of supporting punitive damages award. V. CONCLUSION: Punitive damages are warranted to punish defendant's egregious conduct and deter future fraud. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "financial_law"
        },
        "extra_info": {
            "split": "train",
            "index": 75,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's brief arguing for summary judgment in a defamation case involving online reviews."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR SUMMARY JUDGMENT: DEFAMATION CLAIM: I. INTRODUCTION: Defendant moves for summary judgment because plaintiff cannot establish essential elements of defamation and defendant's statements are protected opinion. II. FACTUAL BACKGROUND: Defendant posted online reviews of plaintiff's [business] expressing dissatisfaction with [services]. The reviews contained defendant's personal opinions about service quality. III. LEGAL STANDARD: Summary judgment is appropriate when no genuine issues of material fact exist. Defamation requires: (1) false statement of fact; (2) publication; (3) damages; (4) fault. IV. ARGUMENT: A. Statements Are Protected Opinion: 1. Subjective Evaluations: Defendant's statements were subjective opinions about service quality, not factual assertions. Phrases like 'poor service' and 'unprofessional' express opinion. 2. Context Indicates Opinion: The statements appeared in consumer review context where readers expect subjective opinions. The setting signals opinion, not fact. B. No False Statement of Fact: 1. Substantially True: To the extent statements contain factual assertions, they are substantially true. Defendant accurately described her experience. 2. Hyperbolic Language: Statements use hyperbolic language typical of opinion expression. Courts recognize such language as non-actionable opinion. C. No Special Damages: Plaintiff cannot prove special damages. General business decline is insufficient without proof of causation. D. Qualified Privilege: Defendant's statements are protected by qualified privilege for consumer reviews. The privilege protects good faith consumer opinions. E. Communications Decency Act: If defendant is not the original author, Section 230 provides immunity for republishing third-party content. V. CONCLUSION: Defendant's statements are protected opinion and summary judgment should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Georgia",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 76,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Section 230"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for temporary restraining order in a trade secrets case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR TEMPORARY RESTRAINING ORDER: I. INTRODUCTION: Plaintiff seeks emergency temporary restraining order to prevent defendant from using and disclosing plaintiff's trade secrets and confidential information. II. FACTUAL BACKGROUND: Defendant, a former employee, had access to plaintiff's trade secrets including [specific information]. Defendant recently joined competitor and is using plaintiff's confidential information. III. LEGAL STANDARD: Temporary restraining orders require: (1) likelihood of success on merits; (2) irreparable harm; (3) balance of hardships favors plaintiff; (4) public interest supports relief. IV. ARGUMENT: A. Likelihood of Success on Merits: 1. Valid Trade Secrets: Plaintiff's [specific information] qualifies as trade secrets because: (a) Information derives economic value from secrecy; (b) Competitors do not know the information; (c) Plaintiff took reasonable steps to maintain secrecy. 2. Misappropriation: Defendant acquired trade secrets through confidential relationship and is using them for competitor's benefit in breach of duty. B. Irreparable Harm: 1. Irreversible Disclosure: Once trade secrets are disclosed, they lose their value permanently. Monetary damages cannot restore secrecy. 2. Competitive Harm: Defendant's use of trade secrets gives competitor unfair advantage. Plaintiff's competitive position is being destroyed. 3. Customer Loss: Defendant is using customer lists and relationships to solicit plaintiff's clients. C. Balance of Hardships: The hardship to plaintiff outweighs inconvenience to defendant. Defendant can compete without using plaintiff's confidential information. D. Public Interest: Enforcing trade secret protection serves public interest by encouraging innovation and investment. E. Emergency Relief Necessary: Each day of delay causes additional irreparable harm. Emergency relief is essential. V. CONCLUSION: Plaintiff respectfully requests immediate temporary restraining order. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "California",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 77,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's brief arguing for dismissal of a qui tam whistleblower action under the False Claims Act."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION TO DISMISS QUI TAM ACTION: I. INTRODUCTION: Defendant moves to dismiss relator's qui tam action because relator lacks standing and fails to plead fraud with particularity. II. LEGAL STANDARD: Rule 9(b) requires fraud allegations to be pleaded with particularity. Qui tam actions require Article III standing and compliance with False Claims Act requirements. III. ARGUMENT: A. Relator Lacks Article III Standing: 1. No Injury in Fact: Relator cannot demonstrate personal injury sufficient for Article III standing. Government's injury is insufficient for relator's standing. 2. Speculation about Government Harm: Relator's allegations about government financial harm are speculative. No concrete evidence shows actual financial loss. B. Failure to Plead Fraud with Particularity: 1. Inadequate Specificity: Complaint fails to identify specific false claims with required detail. Relator must plead who, what, when, where, and how of alleged fraud. 2. Conclusions, Not Facts: Complaint contains conclusory allegations without factual support. Rule 9(b) requires specific factual allegations. C. Public Disclosure Bar: 1. Prior Public Disclosure: The allegations were publicly disclosed in [specific sources] before relator filed suit. This bars qui tam action unless relator is original source. 2. Not Original Source: Relator is not original source of information. Relator learned of allegations from public sources, not independent knowledge. D. Government Knowledge: The government was aware of the alleged conduct through [specific sources]. Government knowledge may bar qui tam action. E. Failure to Serve Government: Relator failed to properly serve government as required by FCA. This jurisdictional defect requires dismissal. IV. CONCLUSION: Relator's qui tam action should be dismissed with prejudice. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "fraud_law"
        },
        "extra_info": {
            "split": "train",
            "index": 78,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Rule 9(b)", "False Claims Act"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's opening statement in a wrongful death case involving a construction accident."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S OPENING STATEMENT: Members of the jury, this case is about a preventable tragedy that took the life of [victim], a loving [husband/father] who was just doing his job. On [date], [victim] was working on a construction site when [specific accident] occurred. He was [age] years old with [family details]. The evidence will show that [victim's] death was caused by [defendant's] negligence and violations of safety regulations. [Defendant] was responsible for maintaining a safe worksite but failed in that duty. The evidence will prove that [defendant]: (1) Violated OSHA safety standards; (2) Failed to provide proper safety equipment; (3) Ignored known hazards; (4) Inadequately trained workers. [Specific safety violations] directly caused this accident. The worksite had [dangerous conditions] that [defendant] knew about but failed to correct. [Victim] was an experienced worker who followed all safety protocols. This accident happened because [defendant] cut corners on safety to save money. The evidence includes: safety inspection reports, witness testimony, expert analysis, and company documents showing [defendant] knew about the hazards. [Victim's] family has suffered an immeasurable loss. [Spouse] lost her husband of [years]. [Children] lost their father who [specific family details]. The economic impact is devastating: [victim] earned $[amount] annually and would have worked [years] more. But no amount of money can replace a life. We seek justice for [victim] and accountability from [defendant]. The evidence will show that [defendant's] negligence caused this tragedy. [Victim] deserved to come home safely to his family. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Texas",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 79,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's brief arguing for exclusion of expert testimony under Daubert standards."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION TO EXCLUDE EXPERT TESTIMONY: I. INTRODUCTION: Defendant moves to exclude plaintiff's expert [name] under Federal Rule of Evidence 702 and Daubert because the proposed testimony is unreliable and not based on sufficient facts. II. LEGAL STANDARD: Rule 702 and Daubert require expert testimony to be: (1) relevant and reliable; (2) based on sufficient facts; (3) product of reliable principles and methods; (4) properly applied to case facts. III. ARGUMENT: A. Unreliable Methodology: 1. Unaccepted Methods: Expert's methodology is not generally accepted in the relevant scientific community. The approach contradicts established [field] principles. 2. Inadequate Testing: Expert failed to test the proposed theory or methodology. The opinions are speculative and lack empirical support. B. Insufficient Factual Basis: 1. Incomplete Investigation: Expert failed to examine critical evidence including [specific evidence]. The analysis is based on incomplete information. 2. Unreliable Assumptions: Expert's conclusions rest on unproven assumptions about [specific issues]. These assumptions lack factual support. C. Irrelevant to Case Issues: 1. Does Not Address Legal Standards: Expert's testimony does not address the relevant legal standards or help resolve disputed issues. 2. Confuses Jury: The testimony will confuse rather than assist the jury in understanding technical issues. D. Qualifications Insufficient: Expert lacks sufficient qualifications in [relevant field]. The expert's background is in [different field] without relevant experience. E. Contradicts Established Science: Expert's opinions contradict well-established scientific principles. The testimony would mislead the jury. IV. CONCLUSION: Plaintiff's expert testimony should be excluded as unreliable and unhelpful. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "evidence_law"
        },
        "extra_info": {
            "split": "train",
            "index": 80,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Federal Rule of Evidence 702", "Daubert"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for certification of a collective action under the Fair Labor Standards Act."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR FLSA COLLECTIVE ACTION CERTIFICATION: I. INTRODUCTION: Plaintiff moves for certification of a collective action under 29 U.S.C. § 216(b) for defendant's systematic violations of the Fair Labor Standards Act. II. LEGAL STANDARD: FLSA collective actions require showing that potential plaintiffs are 'similarly situated' - they suffered similar violations under a common policy or practice. III. FACTUAL BACKGROUND: Defendant classified [number] employees as exempt from overtime despite performing non-exempt duties. All affected employees worked similar jobs under the same policies. IV. ARGUMENT: A. Factual Similarities: 1. Common Job Duties: All potential plaintiffs performed similar non-exempt duties including [specific tasks]. The job descriptions and actual duties were substantially similar. 2. Uniform Misclassification: Defendant applied the same misclassification policy to all potential plaintiffs. The exempt classification was based on job title, not actual duties. B. Similar Legal Claims: 1. Same Legal Theory: All claims arise from defendant's uniform policy of misclassifying employees as exempt. The legal theory is identical for all potential plaintiffs. 2. Common Violations: All potential plaintiffs were denied overtime pay for hours worked over 40 per week. The violations stem from the same policy. C. Common Employer Practices: 1. Centralized Decision-Making: Classification decisions were made centrally by [department/individual]. The same decision-makers applied uniform policies. 2. Systematic Implementation: The misclassification policy was implemented systematically across all locations and departments. D. Efficiency and Judicial Economy: 1. Common Evidence: The same evidence will prove violations for all potential plaintiffs. Individual trials would involve repetitive proof. 2. Avoid Inconsistent Results: Collective resolution prevents inconsistent judgments on identical legal issues. V. CONCLUSION: Certification is appropriate because all potential plaintiffs are similarly situated. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 81,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["29 U.S.C. § 216(b)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's closing argument in a breach of fiduciary duty case involving a business partnership."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S CLOSING ARGUMENT: Members of the jury, this case is about a business relationship that went sour, not a breach of fiduciary duty. My client [name] acted in good faith and in the best interests of the partnership throughout their business relationship. The plaintiff claims my client breached fiduciary duties by [specific allegations]. But the evidence shows my client acted reasonably and within the partnership agreement. Every business decision was made with full knowledge and consent of the partnership. The evidence shows that [client] always disclosed material information to the partnership. All significant decisions were made jointly or with plaintiff's knowledge. There was no self-dealing or secret profits. The plaintiff's expert testified that [client's] conduct was consistent with standard business practices. The partnership agreement specifically authorized [client's] actions. The plaintiff is trying to rewrite the partnership agreement after the fact. When the partnership was profitable, plaintiff never complained about [client's] management. Only when the business faced difficulties did plaintiff claim breach of fiduciary duty. The evidence shows the partnership's problems were caused by [market conditions/external factors], not my client's conduct. My client invested [amount] of personal funds to keep the business operating. This demonstrates good faith, not breach of duty. The plaintiff's damages claim is speculative and unsupported. The business decline was caused by factors beyond my client's control. My client fulfilled all fiduciary obligations and acted in the partnership's best interests. The evidence shows good faith business management, not breach of duty. I ask you to find for my client. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Delaware",
            "legal_domain": "business_law"
        },
        "extra_info": {
            "split": "train",
            "index": 82,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for recovery under a theory of promissory estoppel."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: PROMISSORY ESTOPPEL: I. INTRODUCTION: Plaintiff seeks recovery under promissory estoppel because defendant's promise induced reasonable reliance that caused substantial detriment. II. LEGAL STANDARD: Promissory estoppel requires: (1) clear and definite promise; (2) reasonable reliance; (3) detriment from reliance; (4) injustice can only be avoided by enforcement. III. FACTUAL BACKGROUND: Defendant promised to [specific promise] if plaintiff would [specific action]. Plaintiff reasonably relied on this promise and [specific reliance]. Defendant then refused to perform. IV. ARGUMENT: A. Clear and Definite Promise: 1. Unambiguous Terms: Defendant's promise was clear and definite. The terms were specific enough to create reasonable expectations. 2. Intent to Induce Reliance: Defendant made the promise knowing plaintiff would rely on it. The circumstances show defendant intended to induce reliance. B. Reasonable Reliance: 1. Objective Reasonableness: A reasonable person in plaintiff's position would have relied on defendant's promise. The promise was credible and specific. 2. Actual Reliance: Plaintiff actually relied on the promise by [specific actions]. The reliance was substantial and committed. C. Detriment from Reliance: 1. Substantial Detriment: Plaintiff suffered significant detriment including [specific losses]. The detriment was directly caused by reliance on defendant's promise. 2. Irreversible Changes: Plaintiff made irreversible changes to her position based on the promise. The changes cannot be undone. D. Injustice Requires Enforcement: 1. Unfair Result: Allowing defendant to escape liability would produce an unjust result. Plaintiff reasonably relied to her detriment. 2. No Adequate Alternative: Legal remedies cannot adequately compensate plaintiff's reliance losses. Promissory estoppel is necessary to prevent injustice. V. CONCLUSION: Plaintiff is entitled to recovery under promissory estoppel. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Virginia",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 83,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's brief arguing for summary judgment in a negligence case based on assumption of risk."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR SUMMARY JUDGMENT: ASSUMPTION OF RISK: I. INTRODUCTION: Defendant moves for summary judgment because plaintiff assumed the risk of injury by voluntarily participating in [activity] with knowledge of the dangers. II. LEGAL STANDARD: Assumption of risk bars recovery when plaintiff: (1) knew of the risk; (2) understood the risk; (3) voluntarily assumed the risk. Primary assumption of risk is complete defense. III. FACTUAL BACKGROUND: Plaintiff participated in [activity] and signed waiver acknowledging risks. Plaintiff was experienced in [activity] and knew of inherent dangers. IV. ARGUMENT: A. Plaintiff Knew of Risk: 1. Actual Knowledge: Plaintiff had actual knowledge of the specific risk that caused injury. Plaintiff's deposition testimony confirms knowledge of [specific risk]. 2. Obvious Danger: The risk was open and obvious to any reasonable participant. [Activity] inherently involves [specific dangers]. B. Plaintiff Understood Risk: 1. Experienced Participant: Plaintiff was an experienced participant with [years] of experience in [activity]. This experience provided understanding of risks. 2. Signed Waiver: Plaintiff signed detailed waiver specifically acknowledging [risks]. The waiver demonstrates understanding of dangers. C. Voluntary Assumption: 1. Free Choice: Plaintiff voluntarily chose to participate despite knowing the risks. No coercion or pressure influenced the decision. 2. Continued Participation: Plaintiff continued participating after observing dangerous conditions. This demonstrates voluntary assumption of risk. D. Primary Assumption of Risk: 1. Inherent Risk: The risk that caused injury was inherent in [activity]. Courts recognize [activity] involves inherent dangers. 2. Complete Defense: Primary assumption of risk provides complete defense to negligence claims. No duty exists for inherent risks. V. CONCLUSION: Plaintiff assumed the risk of injury and summary judgment should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Michigan",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 84,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for enforcement of a forum selection clause in a commercial contract."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR ENFORCEMENT OF FORUM SELECTION CLAUSE: I. INTRODUCTION: Plaintiff moves to enforce the forum selection clause in the parties' contract requiring litigation in [jurisdiction]. Defendant must be compelled to litigate in the agreed forum. II. LEGAL STANDARD: Forum selection clauses are presumptively valid and enforceable unless: (1) procurement by fraud or coercion; (2) unreasonable or unjust; (3) contrary to public policy. III. FACTUAL BACKGROUND: The parties' contract contains a clear forum selection clause stating: '[specific clause language].' Defendant initiated litigation in [wrong jurisdiction] in violation of this agreement. IV. ARGUMENT: A. Valid and Enforceable Clause: 1. Clear Language: The forum selection clause is clear and unambiguous. The language specifically designates [jurisdiction] as the exclusive forum for disputes. 2. Mutual Agreement: Both parties agreed to the clause during contract negotiations. The clause was not imposed unilaterally but was mutually negotiated. B. No Fraud or Coercion: 1. Arm's Length Negotiation: The contract was negotiated at arm's length between sophisticated parties. Both parties had legal representation during negotiations. 2. Voluntary Agreement: Defendant voluntarily agreed to the forum selection clause. No evidence exists of fraud, duress, or coercion. C. Reasonable and Just: 1. Reasonable Forum: [Jurisdiction] is a reasonable forum with adequate legal protections. The forum has jurisdiction over the parties and subject matter. 2. No Undue Burden: Litigating in [jurisdiction] does not create undue burden for either party. Both parties contemplated this forum when contracting. D. Public Policy Supports Enforcement: 1. Freedom of Contract: Enforcing forum selection clauses promotes freedom of contract and commercial certainty. 2. Judicial Economy: Enforcement prevents forum shopping and promotes efficient dispute resolution. V. CONCLUSION: The forum selection clause should be enforced and defendant's action dismissed. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 85,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a prosecutor's closing argument in a drug conspiracy case involving multiple defendants."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PROSECUTOR'S CLOSING ARGUMENT: Members of the jury, this case is about a drug conspiracy that brought [drug type] into our community. The evidence proves beyond a reasonable doubt that all defendants participated in a conspiracy to distribute [drug type]. A conspiracy is an agreement between two or more people to commit a crime. The government doesn't need to prove a formal agreement - just that the defendants worked together toward a common goal. The evidence shows defendants [A], [B], and [C] agreed to distribute [drug type] and took steps to carry out that agreement. The evidence includes: [specific evidence - wiretaps, surveillance, witness testimony]. You heard [witness] testify about purchasing drugs from defendant [A]. The surveillance shows defendants meeting at [location] to coordinate drug sales. The wiretaps reveal defendants discussing drug quantities, prices, and distribution methods. Defendant [A] was the leader who organized the conspiracy. The evidence shows [A] coordinated with suppliers, directed other defendants, and controlled territory. Defendant [B] served as a lieutenant, managing street-level dealers. Defendant [C] provided transportation and storage facilities. Each defendant played a crucial role in the conspiracy. The law holds each conspirator responsible for acts of co-conspirators in furtherance of the conspiracy. Each defendant is guilty of the entire conspiracy, not just their individual acts. The defendants will claim they were just friends hanging out. But the evidence shows coordinated criminal activity. The timing of meetings, use of coded language, and exchange of money proves conspiracy. The evidence demands guilty verdicts on all conspiracy counts. Hold these defendants accountable for bringing drugs into our community. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 86,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief arguing for dismissal of a shareholder derivative suit based on demand futility."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION TO DISMISS DERIVATIVE SUIT: I. INTRODUCTION: Defendant moves to dismiss plaintiff's derivative suit because plaintiff failed to make pre-suit demand on the board of directors or adequately plead demand futility. II. LEGAL STANDARD: Derivative suits require either: (1) demand on board of directors; or (2) pleading with particularity that demand would be futile. Demand futility must be evaluated at the time of filing. III. FACTUAL BACKGROUND: Plaintiff filed derivative suit claiming breach of fiduciary duty without making demand on the board. Plaintiff alleges demand futility based on [specific allegations]. IV. ARGUMENT: A. No Pre-Suit Demand: 1. Demand Requirement: Corporate law requires shareholders to demand board action before filing derivative suits. This gives boards opportunity to address alleged wrongdoing. 2. No Demand Made: Plaintiff concedes no demand was made on the board. Plaintiff must therefore prove demand futility. B. Demand Futility Not Established: 1. Independent Directors: The board has [number] independent directors who were not involved in the challenged transaction. These directors could impartially evaluate demand. 2. Business Judgment Protection: The challenged decisions are protected by business judgment rule. The decisions were made by disinterested directors with adequate information. C. Inadequate Pleading: 1. Conclusory Allegations: Plaintiff's demand futility allegations are conclusory and lack specific factual support. Speculation is insufficient to establish futility. 2. Interested Director Standard: Plaintiff must show majority of directors are interested in the challenged transaction. The allegations fail to establish director interest. D. Structural Bias Claims Insufficient: 1. Insufficient Allegations: General allegations about board relationships are insufficient to establish demand futility. Specific facts showing loss of independence required. 2. Normal Business Relationships: The alleged relationships are normal business relationships that don't compromise director independence. V. CONCLUSION: Plaintiff failed to establish demand futility and the derivative suit should be dismissed. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Delaware",
            "legal_domain": "corporate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 87,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for application of the discovery rule in a statute of limitations analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: APPLICATION OF DISCOVERY RULE: I. INTRODUCTION: Plaintiff's claims are timely under the discovery rule because plaintiff could not have reasonably discovered the cause of action until [date], which is within the statute of limitations. II. LEGAL STANDARD: The discovery rule tolls the statute of limitations until the plaintiff knows or reasonably should have known: (1) that an injury occurred; (2) the cause of injury; (3) the potential for legal liability. III. FACTUAL BACKGROUND: Plaintiff suffered [injury] on [date] but did not discover the cause or potential liability until [later date]. The circumstances prevented earlier discovery of the claim. IV. ARGUMENT: A. Injury Not Immediately Apparent: 1. Latent Injury: The injury was not immediately apparent to plaintiff. [Specific injury] developed gradually over time and was not initially diagnosable. 2. Reasonable Lack of Knowledge: A reasonable person in plaintiff's position would not have known of the injury. The symptoms were subtle and non-specific. B. Cause Unknown Until Discovery: 1. Hidden Cause: The cause of injury was hidden from plaintiff. [Specific cause] was not reasonably discoverable through ordinary observation. 2. Professional Diagnosis Required: Discovering the cause required professional medical/expert analysis not available to plaintiff initially. C. Potential Liability Not Apparent: 1. Legal Relationship Unknown: Plaintiff did not know of legal relationship that could give rise to liability. The connection between [defendant] and injury was not apparent. 2. Standard of Care Issues: Plaintiff could not reasonably have known that defendant's conduct fell below applicable standards. D. Reasonable Diligence: 1. Adequate Investigation: Plaintiff exercised reasonable diligence in investigating the injury. All reasonable steps were taken to discover the cause. 2. No Reason for Suspicion: Nothing put plaintiff on notice that further investigation was needed. The circumstances did not suggest legal wrongdoing. V. CONCLUSION: The discovery rule applies and plaintiff's claims are timely filed. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Colorado",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 88,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defense attorney's opening statement in a self-defense case involving a shooting."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENSE OPENING STATEMENT: Members of the jury, this case is about a man who was forced to defend his life. My client [name] is not a criminal - he's a victim who was attacked and had to make a split-second decision to save his own life. The evidence will show that on [date], my client was [location/activity] when [deceased] approached him in a threatening manner. [Deceased] was [size/build comparison] and had a history of violence. The evidence will show that [deceased] initiated the confrontation by [specific actions]. My client tried to avoid the confrontation. He [specific de-escalation attempts]. But [deceased] continued to advance and threatened my client with [specific threats]. My client reasonably believed he was in imminent danger of death or serious bodily harm. The law recognizes the right to self-defense. When someone is attacked, they have the right to protect themselves. My client didn't want to hurt anyone, but he was forced to defend himself. The evidence will show that my client acted reasonably under the circumstances. He was faced with an immediate threat and had no choice but to defend himself. The prosecution wants you to Monday-morning quarterback my client's actions. But the law doesn't require perfect decisions - it requires reasonable decisions under the circumstances. The evidence will show: [specific evidence supporting self-defense]. You'll hear testimony about [deceased's] violent history and threatening behavior. The physical evidence confirms my client's account. This is a tragedy, but it's not a crime. My client exercised his fundamental right to self-defense. The evidence will show he acted lawfully and reasonably. Thank you.",
            "task_type": "opinion_generation",
            "jurisdiction": "Florida",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 89,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for rescission of a contract based on mutual mistake."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR RESCISSION BASED ON MUTUAL MISTAKE: I. INTRODUCTION: Plaintiff seeks rescission of contract because both parties operated under a mutual mistake about a basic assumption that materially affected the agreement. II. LEGAL STANDARD: Rescission for mutual mistake requires: (1) mistake by both parties; (2) relates to basic assumption of contract; (3) material effect on agreed exchange; (4) party seeking rescission did not bear risk of mistake. III. FACTUAL BACKGROUND: Both parties believed [specific mistaken fact] when entering the contract. This belief was fundamental to the agreement. Subsequent discovery revealed the true facts. IV. ARGUMENT: A. Mutual Mistake by Both Parties: 1. Shared Belief: Both parties shared the mistaken belief about [specific fact]. The evidence shows neither party knew the true facts. 2. Reasonable Mistake: The mistake was reasonable given the available information. Both parties conducted reasonable investigation before contracting. B. Basic Assumption of Contract: 1. Fundamental to Agreement: The mistaken belief related to a basic assumption underlying the contract. The parties would not have agreed without this assumption. 2. Essential Element: The mistake concerned an essential element of the contract's subject matter. The assumption was central to the bargain. C. Material Effect on Exchange: 1. Significant Impact: The mistake materially affects the agreed exchange. The true facts make the contract substantially different from what parties intended. 2. Altered Performance: The mistake makes performance substantially more burdensome or less valuable than anticipated. D. Risk Allocation: 1. Risk Not Allocated: The contract did not allocate risk of this mistake to either party. Neither party assumed the risk of the mistaken assumption. 2. Unforeseen Circumstance: The mistake involved circumstances that were unforeseen and unforeseeable at contract formation. E. Equitable Relief Appropriate: Rescission is equitable because both parties were mistaken and enforcement would be unfair. V. CONCLUSION: Rescission is appropriate based on mutual mistake. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Virginia",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 90,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for a constructive discharge claim in an employment discrimination case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: CONSTRUCTIVE DISCHARGE CLAIM: I. INTRODUCTION: Plaintiff was constructively discharged when defendant created intolerable working conditions that forced resignation. The working conditions were so severe that a reasonable person would have felt compelled to resign. II. LEGAL STANDARD: Constructive discharge requires: (1) reasonable person would find conditions intolerable; (2) conditions were intended to force resignation or were result of discrimination; (3) plaintiff had no reasonable alternative but to resign. III. FACTUAL BACKGROUND: After plaintiff filed discrimination complaint, defendant subjected plaintiff to: [specific retaliatory actions]. The harassment escalated until plaintiff had no choice but to resign. IV. ARGUMENT: A. Objectively Intolerable Conditions: 1. Severe Harassment: Defendant subjected plaintiff to constant harassment including [specific acts]. The conduct was severe and pervasive. 2. Reasonable Person Standard: A reasonable person in plaintiff's position would have found the conditions intolerable. The harassment made effective job performance impossible. B. Discriminatory Intent: 1. Retaliatory Motive: The harassment began after plaintiff filed discrimination complaint. The timing shows retaliatory intent. 2. Pattern of Discrimination: The conduct was part of broader pattern of discrimination against [protected class]. Other employees received similar treatment. C. No Reasonable Alternative: 1. Inadequate Internal Remedies: Plaintiff reported harassment to management but no effective action was taken. The company failed to stop the harassment. 2. Escalating Conduct: The harassment was escalating and showed no signs of stopping. Plaintiff reasonably believed conditions would worsen. D. Constructive Discharge Doctrine: 1. Forced Resignation: Defendant's conduct forced plaintiff to resign. The resignation was involuntary despite formal appearance of quitting. 2. Circumventing Protections: Constructive discharge doctrine prevents employers from circumventing employment protections by forcing resignations. V. CONCLUSION: Plaintiff was constructively discharged and entitled to relief. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 91,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief arguing for summary judgment in a premises liability case involving a slip and fall."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR SUMMARY JUDGMENT: PREMISES LIABILITY: I. INTRODUCTION: Defendant moves for summary judgment because plaintiff cannot establish that defendant had notice of the dangerous condition or that defendant breached its duty of care. II. LEGAL STANDARD: Premises liability requires: (1) duty of care; (2) breach of duty; (3) causation; (4) damages. Property owners must have actual or constructive notice of dangerous conditions. III. FACTUAL BACKGROUND: Plaintiff slipped on [substance] in defendant's store. Plaintiff claims defendant should have known about the condition and cleaned it up. IV. ARGUMENT: A. No Notice of Dangerous Condition: 1. No Actual Notice: Defendant had no actual notice of the spill. No employee witnessed the spill or received reports about it. 2. No Constructive Notice: Plaintiff cannot establish constructive notice. The substance was not on the floor long enough for defendant to discover it through reasonable inspection. B. Reasonable Care in Maintenance: 1. Adequate Inspection: Defendant conducted reasonable inspections of the premises. Employee testimony shows regular cleaning and inspection procedures. 2. Proper Procedures: Defendant had proper procedures for identifying and cleaning spills. The procedures met industry standards. C. Open and Obvious Doctrine: 1. Obvious Danger: The condition was open and obvious to plaintiff. A reasonable person would have seen and avoided the hazard. 2. Comparative Negligence: Plaintiff's own negligence contributed to the accident. Plaintiff failed to watch where walking. D. Causation Issues: 1. Alternative Causes: The fall may have been caused by factors other than the alleged dangerous condition. Plaintiff's medical conditions could have contributed. 2. Insufficient Evidence: Plaintiff cannot establish that the condition caused the fall. E. No Duty to Continuously Clean: Property owners have no duty to continuously clean or maintain premises. Reasonable care under the circumstances is sufficient. V. CONCLUSION: Summary judgment should be granted because plaintiff cannot establish essential elements. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "California",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 92,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for disgorgement of profits in a trademark infringement case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR DISGORGEMENT OF PROFITS: I. INTRODUCTION: Plaintiff seeks disgorgement of defendant's profits from trademark infringement because defendant willfully infringed plaintiff's trademark and profited from consumer confusion. II. LEGAL STANDARD: Disgorgement requires: (1) trademark infringement; (2) defendant's profits attributable to infringement; (3) profits reasonably related to infringement; (4) consideration of willfulness and bad faith. III. FACTUAL BACKGROUND: Defendant used confusingly similar mark [defendant's mark] that infringed plaintiff's registered trademark [plaintiff's mark]. Defendant's sales increased significantly after adopting the infringing mark. IV. ARGUMENT: A. Established Trademark Infringement: 1. Valid Trademark: Plaintiff owns valid registered trademark [mark] for [goods/services]. The mark is distinctive and protectable. 2. Likelihood of Confusion: Defendant's mark creates likelihood of confusion with plaintiff's mark. Consumer survey evidence shows actual confusion. B. Defendant's Profits from Infringement: 1. Increased Sales: Defendant's sales increased [percentage] after adopting infringing mark. Financial records show correlation between mark adoption and sales growth. 2. Causal Connection: The increased sales were caused by consumer confusion. Consumers bought defendant's products believing they were plaintiff's. C. Willful Infringement: 1. Knowledge of Plaintiff's Mark: Defendant knew of plaintiff's trademark before adopting similar mark. Evidence shows defendant researched plaintiff's mark. 2. Intentional Copying: Defendant intentionally copied distinctive elements of plaintiff's mark. The similarity was deliberate, not coincidental. D. Appropriate Remedy: 1. Deterrent Effect: Disgorgement serves important deterrent function. Allowing infringers to keep profits encourages infringement. 2. Unjust Enrichment: Defendant was unjustly enriched by using plaintiff's trademark goodwill. Disgorgement prevents unjust enrichment. E. Calculation of Profits: 1. Defendant's Burden: Defendant must prove expenses and deductions. Plaintiff need only prove defendant's gross revenues. 2. Reasonable Relationship: All profits are reasonably related to infringement because defendant's entire business benefited from confused consumers. V. CONCLUSION: Disgorgement of profits is appropriate to remedy defendant's willful infringement. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 93,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defense attorney's motion for acquittal in a criminal case based on insufficient evidence."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR JUDGMENT OF ACQUITTAL: I. INTRODUCTION: Defendant moves for judgment of acquittal under Rule 29 because the prosecution failed to present sufficient evidence to support conviction on any count. II. LEGAL STANDARD: Judgment of acquittal is appropriate when no reasonable jury could find defendant guilty beyond a reasonable doubt. Evidence must be viewed in light most favorable to prosecution. III. ARGUMENT: A. Insufficient Evidence of [Specific Element]: 1. Prosecution's Burden: Prosecution must prove every element of the offense beyond a reasonable doubt. The evidence fails to establish [specific element]. 2. Speculation, Not Evidence: Prosecution relies on speculation and conjecture rather than evidence. Gaps in evidence cannot be filled by assumption. B. Lack of Evidence on Intent: 1. No Evidence of Criminal Intent: Prosecution presented no evidence that defendant acted with required criminal intent. The conduct has innocent explanations. 2. Circumstantial Evidence Insufficient: Circumstantial evidence does not support inference of criminal intent. Alternative explanations are equally reasonable. C. Credibility Issues: 1. Unreliable Witnesses: Prosecution's key witnesses have credibility problems including [specific issues]. Their testimony is inherently unreliable. 2. Inconsistent Testimony: Witness testimony is internally inconsistent and contradicted by physical evidence. D. Burden of Proof Not Met: 1. Reasonable Doubt: The evidence creates reasonable doubt about defendant's guilt. Alternative explanations are consistent with evidence. 2. Mere Suspicion: Prosecution evidence creates suspicion but not proof beyond reasonable doubt. Suspicion is insufficient for conviction. E. Constitutional Protections: 1. Due Process: Conviction on insufficient evidence violates due process. Courts must ensure adequate evidence supports verdicts. 2. Burden on Prosecution: Prosecution bears burden to prove guilt. Defendant has no burden to prove innocence. IV. CONCLUSION: The evidence is insufficient to support conviction and judgment of acquittal should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Pennsylvania",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 94,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Rule 29"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for equitable relief in a breach of covenant not to compete case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR EQUITABLE RELIEF: NON-COMPETE VIOLATION: I. INTRODUCTION: Plaintiff seeks injunctive relief to enforce defendant's covenant not to compete because defendant's violation threatens irreparable harm to plaintiff's business. II. LEGAL STANDARD: Injunctive relief requires: (1) likelihood of success on merits; (2) irreparable harm; (3) balance of hardships favors plaintiff; (4) public interest supports relief. III. FACTUAL BACKGROUND: Defendant signed non-compete agreement prohibiting [specific restrictions] for [duration] in [geographic area]. Defendant violated the agreement by [specific violations]. IV. ARGUMENT: A. Likelihood of Success on Merits: 1. Valid Covenant: The non-compete agreement is valid and enforceable. It protects legitimate business interests including [customer relationships, trade secrets, specialized training]. 2. Reasonable Restrictions: The restrictions are reasonable in scope, duration, and geographic area. They are narrowly tailored to protect plaintiff's interests. 3. Clear Breach: Defendant clearly breached the agreement by [specific conduct]. The breach is ongoing and continuing. B. Irreparable Harm: 1. Loss of Trade Secrets: Defendant's breach threatens disclosure of trade secrets and confidential information. This harm cannot be compensated by monetary damages. 2. Customer Relationships: Defendant is using relationships developed during employment to compete unfairly. Loss of customers causes irreparable harm. 3. Competitive Advantage: Defendant's breach gives unfair competitive advantage that damages plaintiff's market position. C. Balance of Hardships: 1. Plaintiff's Greater Harm: Plaintiff faces substantial harm from continuing breach. Business relationships and competitive position are threatened. 2. Defendant's Self-Imposed Hardship: Defendant voluntarily agreed to restrictions and chose to breach. Any hardship is self-imposed. D. Public Interest: 1. Enforcement of Contracts: Public interest supports enforcement of valid contracts. Parties should be held to their agreements. 2. Fair Competition: Injunction promotes fair competition by preventing use of confidential information and trade secrets. E. Appropriate Scope: Injunctive relief should be limited to terms of the agreement. Court should enforce reasonable restrictions as written. V. CONCLUSION: Injunctive relief is appropriate to prevent continuing breach and protect plaintiff's interests. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Texas",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 95,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a defendant's brief arguing for limitation of damages in a contract breach case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S BRIEF: LIMITATION OF DAMAGES: I. INTRODUCTION: Defendant seeks limitation of damages because plaintiff's claimed damages are not foreseeable, fail to satisfy the certainty requirement, and were not properly mitigated. II. LEGAL STANDARD: Contract damages must be: (1) foreseeable at contract formation; (2) reasonably certain; (3) caused by breach; (4) properly mitigated. Consequential damages require special foreseeability. III. FACTUAL BACKGROUND: Plaintiff claims [amount] in damages from defendant's breach. The claimed damages include lost profits, consequential damages, and incidental costs. IV. ARGUMENT: A. Lack of Foreseeability: 1. Hadley v. Baxendale Standard: Damages must be foreseeable at contract formation. Plaintiff's extraordinary losses were not reasonably foreseeable. 2. No Special Circumstances: Defendant was not informed of special circumstances that would make unusual damages foreseeable. The contract was standard commercial transaction. B. Uncertainty of Lost Profits: 1. Speculative Damages: Plaintiff's lost profit calculations are speculative and uncertain. New business profits are particularly uncertain. 2. Insufficient Evidence: Plaintiff cannot establish lost profits with reasonable certainty. The evidence is based on projections and assumptions. C. Failure to Mitigate: 1. Mitigation Requirement: Plaintiff had duty to mitigate damages but failed to take reasonable steps. Alternative suppliers were available. 2. Increased Damages: Plaintiff's actions increased rather than mitigated damages. The failure to mitigate bars recovery of avoidable damages. D. Causation Issues: 1. Intervening Causes: Plaintiff's damages were caused by market conditions, not defendant's breach. Independent factors contributed to losses. 2. Proximate Cause: Defendant's breach was not proximate cause of claimed damages. The causal chain was broken by intervening events. E. Limitation of Liability Clause: 1. Valid Limitation: Contract contains valid limitation of liability clause. The clause should be enforced according to its terms. 2. Unconscionability Challenge: Plaintiff's unconscionability challenge fails. The clause was negotiated between sophisticated parties. F. Incidental vs. Consequential: Many claimed damages are incidental rather than consequential. Incidental damages are subject to different rules and limitations. V. CONCLUSION: Damages should be limited to foreseeable, certain losses that were properly mitigated. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Illinois",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 96,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Hadley v. Baxendale"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a plaintiff's brief arguing for joint and several liability in a multi-defendant tort case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF: JOINT AND SEVERAL LIABILITY: I. INTRODUCTION: Plaintiff seeks application of joint and several liability because defendants acted in concert to cause indivisible harm, making each defendant fully liable for plaintiff's damages. II. LEGAL STANDARD: Joint and several liability applies when: (1) defendants act in concert; (2) harm is indivisible; (3) each defendant's conduct is factual cause of harm; (4) jurisdiction recognizes joint and several liability. III. FACTUAL BACKGROUND: Defendants [A], [B], and [C] each contributed to causing plaintiff's [injury]. The harm resulted from defendants' combined conduct and cannot be apportioned among them. IV. ARGUMENT: A. Concerted Action: 1. Common Plan: Defendants acted pursuant to common plan or design. Evidence shows coordination and agreement among defendants. 2. Encouragement and Assistance: Each defendant encouraged and assisted the others in tortious conduct. The conduct was collaborative, not independent. B. Indivisible Harm: 1. Single Injury: Plaintiff suffered single, indivisible injury that cannot be apportioned among defendants. The harm is unified and cannot be separated. 2. Burden on Defendants: Defendants bear burden of proving their conduct caused only divisible portion of harm. They cannot meet this burden. C. Factual Causation: 1. But-For Causation: Each defendant's conduct was but-for cause of plaintiff's harm. Removing any defendant's conduct would have prevented the injury. 2. Substantial Factor: Each defendant's conduct was substantial factor in causing harm. All defendants materially contributed to the result. D. Policy Considerations: 1. Plaintiff Protection: Joint and several liability protects injured plaintiff from insolvent defendants. Plaintiff should not bear risk of defendant insolvency. 2. Deterrent Effect: Joint liability encourages careful conduct by making each defendant fully responsible for consequences. E. Comparative Fault Considerations: 1. Fault Among Defendants: Any comparative fault analysis should be among defendants, not against plaintiff. Plaintiff should recover full damages. 2. Contribution Rights: Defendants retain contribution rights against co-defendants. Joint and several liability does not eliminate contribution. F. Jurisdictional Law: [Jurisdiction] recognizes joint and several liability for concerted tortious conduct. The doctrine applies to these facts. V. CONCLUSION: Joint and several liability should apply, making each defendant fully liable for plaintiff's damages. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "New_York",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 97,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a prosecutor's brief opposing a motion to suppress evidence in a search warrant case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PROSECUTOR'S BRIEF OPPOSING MOTION TO SUPPRESS: I. INTRODUCTION: Defendant's motion to suppress should be denied because the search warrant was supported by probable cause and executed in reasonable manner. The evidence was lawfully obtained. II. LEGAL STANDARD: Search warrants require probable cause supported by oath or affirmation. Evidence should be suppressed only if Fourth Amendment was violated and exclusion serves deterrent purpose. III. FACTUAL BACKGROUND: Police obtained search warrant for defendant's residence based on [specific probable cause]. The warrant was executed on [date] and evidence of [crimes] was discovered. IV. ARGUMENT: A. Probable Cause Established: 1. Reliable Information: Warrant affidavit contained reliable information from [confidential informant/surveillance/other sources]. The information was corroborated by independent investigation. 2. Nexus to Location: Evidence established nexus between criminal activity and defendant's residence. The connection was reasonable and supported by facts. B. Valid Warrant Execution: 1. Proper Notice: Officers provided proper notice before entering. The knock-and-announce requirement was satisfied. 2. Scope of Search: Officers stayed within warrant's scope. Only authorized areas were searched and only specified items were seized. C. Good Faith Exception: 1. Reasonable Reliance: Even if warrant was defective, officers reasonably relied on magistrate's determination. The good faith exception applies. 2. No Deterrent Effect: Suppression would not deter police misconduct because officers acted reasonably. Exclusion serves no purpose. D. Inevitable Discovery: 1. Independent Investigation: Evidence would have been discovered through independent investigation. Defendant was under surveillance independent of warrant. 2. Routine Procedures: Evidence would have been discovered through routine booking procedures. Alternative discovery was certain. E. Attenuation Doctrine: 1. Intervening Acts: Any Fourth Amendment violation was attenuated by intervening acts including [specific acts]. The taint was purged. 2. Voluntary Conduct: Defendant's voluntary conduct broke causal chain between any violation and evidence discovery. F. Standing Issues: Defendant lacks standing to challenge certain evidence because [specific standing issues]. Only evidence from areas where defendant had reasonable expectation of privacy can be suppressed. V. CONCLUSION: The search was constitutional and defendant's motion should be denied. Respectfully submitted, [Attorney for State].",
            "task_type": "opinion_generation",
            "jurisdiction": "North_Carolina",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 98,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Prepare a defendant's brief arguing for summary judgment in a negligent supervision case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "DEFENDANT'S MOTION FOR SUMMARY JUDGMENT: NEGLIGENT SUPERVISION: I. INTRODUCTION: Defendant moves for summary judgment because plaintiff cannot establish that defendant had duty to supervise or that defendant breached any supervisory duty. II. LEGAL STANDARD: Negligent supervision requires: (1) duty to supervise; (2) breach of supervisory duty; (3) causation; (4) damages. The duty must be based on special relationship or statutory obligation. III. FACTUAL BACKGROUND: Plaintiff claims defendant negligently supervised [person/entity] who caused plaintiff's injury. Defendant argues no supervisory duty existed. IV. ARGUMENT: A. No Duty to Supervise: 1. No Special Relationship: Defendant had no special relationship with [person] that created duty to supervise. The relationship was [employment/business/other] without supervisory obligations. 2. Independent Contractor: [Person] was independent contractor, not employee. Defendant had no right to control manner of work performance. B. No Breach of Supervisory Duty: 1. Reasonable Supervision: Defendant provided reasonable supervision consistent with industry standards. The supervision was adequate for the circumstances. 2. Proper Procedures: Defendant had proper procedures for supervision including [specific procedures]. The procedures were followed. C. Lack of Foreseeability: 1. Unforeseeable Conduct: The harmful conduct was not foreseeable based on [person's] history and behavior. Defendant had no reason to expect dangerous conduct. 2. No Prior Notice: Defendant had no prior notice of [person's] dangerous propensities. The conduct was unprecedented. D. Causation Issues: 1. Superseding Cause: [Person's] conduct was superseding cause that breaks causal chain. The conduct was independent criminal act. 2. No Proximate Cause: Even if supervisory duty existed, breach was not proximate cause of plaintiff's injury. Intervening factors caused harm. E. Adequate Training and Screening: 1. Proper Hiring: Defendant conducted proper background checks and screening. [Person] was qualified and had no history of problems. 2. Adequate Training: Defendant provided adequate training for [person's] duties. The training met industry standards. F. Statutory Immunity: Defendant may be entitled to statutory immunity under [relevant statute]. The immunity protects against negligent supervision claims. V. CONCLUSION: Defendant had no duty to supervise and provided adequate supervision. Summary judgment should be granted. Respectfully submitted, [Attorney for Defendant].",
            "task_type": "opinion_generation",
            "jurisdiction": "Florida",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 99,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "Draft a plaintiff's brief arguing for emergency relief in a domestic violence protection order case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "PLAINTIFF'S BRIEF FOR EMERGENCY PROTECTION ORDER: I. INTRODUCTION: Plaintiff seeks emergency protection order because defendant's escalating domestic violence threatens imminent harm to plaintiff and [children]. Emergency relief is necessary to prevent further abuse. II. LEGAL STANDARD: Emergency protection orders require: (1) evidence of domestic violence; (2) imminent threat of harm; (3) emergency circumstances; (4) no adequate alternative remedies. III. FACTUAL BACKGROUND: Defendant has subjected plaintiff to pattern of domestic violence including [specific incidents]. The violence has escalated and defendant recently threatened [specific threats]. IV. ARGUMENT: A. Evidence of Domestic Violence: 1. Pattern of Abuse: Defendant subjected plaintiff to systematic domestic violence over [timeframe]. The abuse included physical violence, threats, and emotional abuse. 2. Documented Incidents: Medical records, police reports, and photographs document defendant's violent conduct. The evidence shows escalating pattern. B. Imminent Threat of Harm: 1. Recent Threats: Defendant recently threatened to [specific threats]. The threats were specific and credible. 2. Escalating Violence: Defendant's violence has escalated in frequency and severity. The pattern shows increasing danger. 3. Access to Weapons: Defendant has access to [weapons] and has threatened to use them. The weapons increase the danger. C. Emergency Circumstances: 1. Immediate Danger: Plaintiff faces immediate danger from defendant's threats and violent behavior. Delay in relief could result in serious harm. 2. No Time for Notice: Emergency circumstances justify ex parte relief. Providing notice to defendant would increase danger. D. No Alternative Remedies: 1. Inadequate Police Response: Police intervention has been inadequate to stop defendant's violence. Criminal charges are pending but provide no immediate protection. 2. Shelter Limitations: Shelter space is limited and does not provide long-term solution. Protection order is necessary for plaintiff's safety. E. Children's Safety: 1. Witnessed Violence: Children witnessed domestic violence and are traumatized. They need protection from further exposure. 2. Threats to Children: Defendant threatened to harm children. Their safety requires immediate protection. F. Requested Relief: 1. No Contact Order: Plaintiff requests order prohibiting defendant from contacting plaintiff or children. 2. Residence Protection: Defendant should be excluded from family residence. 3. Temporary Custody: Plaintiff seeks temporary custody of children with supervised visitation only. V. CONCLUSION: Emergency protection order is necessary to prevent imminent harm. Respectfully submitted, [Attorney for Plaintiff].",
            "task_type": "opinion_generation",
            "jurisdiction": "Arizona",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 100,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm representing a client in a California state court products liability case. The opposing counsel cited Miranda v. Arizona (1966) in their brief. Is this precedent controlling, persuasive, or irrelevant to my case?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: Miranda v. Arizona is **irrelevant** to your products liability case. Precedent hierarchy analysis: (1) **Subject Matter Mismatch**: Miranda established constitutional protections for criminal suspects during custodial interrogation, while your case involves civil products liability law. (2) **Jurisdictional Authority**: Although Miranda is Supreme Court precedent that would be controlling on constitutional criminal procedure issues, it has no bearing on state tort law. (3) **Legal Domain Separation**: Criminal procedure precedents do not control civil tort claims unless constitutional issues are involved. (4) **Appropriate Response**: You should file a motion to strike or object to the citation as irrelevant and potentially misleading to the court. (5) **Relevant Precedent Sources**: Focus instead on California products liability cases, federal circuit court decisions on similar products, and Supreme Court precedents on products liability or tort law. (6) **Strategic Consideration**: The irrelevant citation may indicate opposing counsel's weak understanding of applicable law or attempt to confuse the issues. This strengthens your position if you can demonstrate command of actual controlling authority.",
            "task_type": "precedent_analysis",
            "jurisdiction": "California",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 101,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Miranda v. Arizona (1966)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm arguing a contract interpretation case in New York state court. The opposing party cited a Second Circuit federal decision on contract interpretation. How should I analyze this precedent's authority?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: The Second Circuit decision is **persuasive authority** with moderate influence. Hierarchical analysis: (1) **Jurisdictional Relationship**: Second Circuit covers New York federal courts, creating geographical overlap but different court systems. (2) **Persuasive Value**: While not binding on New York state courts, Second Circuit decisions carry significant persuasive weight due to: (a) Geographic jurisdiction over New York; (b) Federal judges' expertise; (c) Thorough analysis typical of appellate decisions. (3) **Contract Law Considerations**: If the case involves federal contract law (federal contracts, interstate commerce), the precedent is stronger. If purely state contract law, it's weaker but still relevant. (4) **Strategic Response Options**: (a) **Distinguish** if unfavorable - show factual or legal differences; (b) **Embrace** if favorable - argue persuasive value and sound reasoning; (c) **Counter** with New York state precedents that conflict. (5) **Precedent Hierarchy**: New York Court of Appeals > NY Appellate Division > Second Circuit > other federal circuits > other state courts. (6) **Research Strategy**: Find New York state cases on point, as they would be controlling over federal persuasive authority.",
            "task_type": "precedent_analysis",
            "jurisdiction": "New_York",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 102,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in federal district court in Texas on a First Amendment case. The Fifth Circuit just decided a similar case last month, but the Supreme Court decided a related case 10 years ago. How do I analyze these precedents?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: Both precedents are **controlling authority** with different hierarchical weight. Authority analysis: (1) **Supreme Court Precedent** (10 years old): **Highest controlling authority** - Supreme Court decisions bind all lower courts on constitutional issues. Age doesn't diminish binding effect unless overruled. Must analyze: (a) Exact constitutional holding; (b) Whether subsequent decisions have modified the rule; (c) Factual distinctions from your case. (2) **Fifth Circuit Precedent** (recent): **Directly controlling** - Fifth Circuit decisions bind all district courts within the circuit. Recent timing indicates: (a) Current judicial interpretation; (b) Application of Supreme Court precedent to similar facts; (c) Likely reflects most recent constitutional understanding. (3) **Hierarchical Relationship**: Supreme Court precedent establishes broad constitutional framework, Fifth Circuit precedent applies that framework to specific circumstances. (4) **Analysis Strategy**: (a) Start with Supreme Court precedent for fundamental constitutional principles; (b) Apply Fifth Circuit precedent for specific application and current interpretation; (c) Argue how both precedents support your position. (5) **Potential Conflicts**: If they conflict, Supreme Court precedent controls, but seek to harmonize both decisions. (6) **Strategic Advantage**: Recent Fifth Circuit case likely considered the older Supreme Court precedent, showing evolution of jurisprudence.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 103,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching an employment discrimination case in Illinois. I found a very similar case from the Seventh Circuit, but also a contradictory decision from the Illinois Supreme Court. Which precedent should control?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Depends on the legal claims involved** - federal vs. state law determines controlling authority. Jurisdiction analysis: (1) **Federal Employment Claims** (Title VII, ADEA, ADA): **Seventh Circuit controls** - Federal courts have exclusive/primary jurisdiction over federal employment statutes. Illinois Supreme Court decision would be irrelevant unless it involves state constitutional interpretation. (2) **State Employment Claims** (Illinois Human Rights Act, state tort claims): **Illinois Supreme Court controls** - State supreme courts are final arbiters of state law. Seventh Circuit decision would be persuasive only. (3) **Mixed Claims** (both federal and state): **Dual analysis required** - Apply Seventh Circuit precedent to federal claims, Illinois Supreme Court to state claims. Look for: (a) Overlapping legal principles; (b) Factual distinctions; (c) Policy considerations. (4) **Conflict Resolution**: If both precedents apply to same issue: (a) **Federal preemption** - Federal law may preempt state law; (b) **Separate standards** - Different legal standards may apply; (c) **Supplemental jurisdiction** - Federal court may need to apply state law. (5) **Strategic Considerations**: (a) **Claim selection** - Choose claims that invoke favorable precedent; (b) **Forum shopping** - Consider whether federal or state court is preferable; (c) **Precedent timing** - Recent decisions may indicate trend. (6) **Research Priority**: Determine exact legal basis for your claims before prioritizing precedent authority.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Illinois",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 104,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm arguing a criminal case in Florida state court. The prosecutor cited a case from the Eleventh Circuit. The defense cited a Florida Supreme Court case that seems to contradict it. How do I analyze this precedent conflict?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Florida Supreme Court precedent controls** on state criminal law issues. Conflict resolution analysis: (1) **Jurisdictional Hierarchy**: Florida Supreme Court is highest authority on Florida state criminal law. Eleventh Circuit precedent would only be controlling if it involves federal constitutional issues or federal crimes. (2) **Subject Matter Analysis**: (a) **State criminal law** - Florida Supreme Court controls; (b) **Federal constitutional criminal procedure** - Eleventh Circuit precedent has strong persuasive value but Florida Supreme Court can provide greater protection; (c) **Federal crimes** - Eleventh Circuit would control. (3) **Constitutional Floor Principle**: Federal precedents establish minimum constitutional protections, but state courts can provide greater protections under state constitutions. (4) **Strategic Response**: (a) **Emphasize** Florida Supreme Court precedent as controlling authority; (b) **Distinguish** Eleventh Circuit case on facts or federal vs. state law basis; (c) **Argue** state constitutional protections exceed federal minimums if applicable. (5) **Precedent Strength Assessment**: (a) **Recency** - More recent decisions carry more weight; (b) **Factual similarity** - Cases with similar facts are more persuasive; (c) **Reasoning quality** - Well-reasoned decisions have greater influence. (6) **Research Strategy**: Find additional Florida precedents supporting your position, as they would strengthen your argument against federal persuasive authority.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Florida",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 105,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a patent case in federal court. There's a Federal Circuit decision on point, but also a more recent district court decision from another circuit that disagrees. Which precedent has more weight?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Federal Circuit precedent is controlling** and significantly outweighs the district court decision. Patent law hierarchy analysis: (1) **Federal Circuit Authority**: Federal Circuit has exclusive appellate jurisdiction over patent cases, making its decisions controlling nationwide on patent law issues. Its precedents bind all district courts. (2) **District Court Precedent**: District court decisions are **not binding** on other district courts, even within the same circuit. They are merely persuasive authority with limited weight. (3) **Hierarchical Comparison**: Federal Circuit >> District Court from any circuit. The district court decision cannot override Federal Circuit precedent. (4) **Strategic Considerations**: (a) **Follow Federal Circuit precedent** - It's controlling law; (b) **Distinguish district court case** - Show factual or legal differences; (c) **Argue for Federal Circuit review** - If district court created circuit split, it may warrant Federal Circuit attention. (5) **Precedent Analysis**: (a) **Federal Circuit reasoning** - Examine rationale and scope of holding; (b) **District court criticism** - Understand why it disagreed, but note it cannot override controlling precedent; (c) **Future implications** - District court disagreement might signal need for Federal Circuit clarification. (6) **Litigation Strategy**: Emphasize Federal Circuit precedent as controlling, use district court decision only if it provides helpful reasoning that doesn't contradict Federal Circuit holding.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 106,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I found a Supreme Court case from 1985 that favors my position, but the Ninth Circuit decided a case last year that seems to narrow its application. How do I analyze the continuing vitality of the older precedent?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Supreme Court precedent remains controlling** but requires careful analysis of its current scope. Vitality assessment: (1) **Hierarchical Authority**: Supreme Court precedent (1985) is controlling over Ninth Circuit precedent (recent). The Ninth Circuit cannot overrule Supreme Court precedent, only interpret its application. (2) **Precedent Evolution Analysis**: (a) **Explicit overruling** - Has Supreme Court explicitly overruled or modified the 1985 decision? (b) **Implicit limitation** - Have subsequent Supreme Court cases narrowed its scope? (c) **Circuit interpretation** - Is Ninth Circuit properly applying or improperly restricting the precedent? (3) **Narrowing vs. Distinguishing**: (a) **Proper narrowing** - Ninth Circuit may legitimately apply precedent to different factual circumstances; (b) **Improper restriction** - Circuit cannot ignore or contradict Supreme Court holding. (4) **Research Strategy**: (a) **Shepardize/KeyCite** the 1985 case for subsequent Supreme Court treatment; (b) **Analyze intervening precedent** - What has Supreme Court said since 1985?; (c) **Examine circuit split** - Do other circuits agree with Ninth Circuit interpretation? (5) **Litigation Approach**: (a) **Emphasize Supreme Court precedent** as controlling; (b) **Distinguish Ninth Circuit case** on facts or scope; (c) **Argue proper interpretation** of Supreme Court precedent. (6) **Strategic Considerations**: If Ninth Circuit improperly restricted Supreme Court precedent, this could be grounds for Supreme Court review.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 107,
            "difficulty": "advanced",
            "court_level": "federal_circuit",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a Delaware Chancery Court case involving corporate governance. There's a Delaware Supreme Court case from 2019 on point, but also a well-reasoned Delaware Chancery Court decision from 2023 that takes a different approach. How do I evaluate these precedents?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Delaware Supreme Court precedent (2019) is controlling** despite the more recent Chancery Court decision. Delaware corporate law hierarchy: (1) **Controlling Authority**: Delaware Supreme Court decisions are binding on all Delaware courts, including Chancery Court. The 2019 decision controls unless: (a) It's been overruled; (b) The legal landscape has changed significantly; (c) The facts are materially distinguishable. (2) **Chancery Court Decision Analysis**: The 2023 decision is **not binding** but carries significant persuasive weight because: (a) **Specialized expertise** - Chancery Court judges are corporate law specialists; (b) **Recent analysis** - May reflect evolving business practices; (c) **Detailed reasoning** - Well-reasoned decisions influence future law. (3) **Potential Explanations**: (a) **Factual distinction** - 2023 case may involve different facts warranting different result; (b) **Legal evolution** - Intervening statutes or precedents may have changed analysis; (c) **Erroneous application** - Chancery Court may have misapplied Supreme Court precedent. (4) **Strategic Approach**: (a) **Primary reliance** on Supreme Court precedent; (b) **Distinguish** Chancery Court case if unfavorable; (c) **Harmonize** both decisions if possible; (d) **Appeal potential** - If Chancery Court incorrectly applied Supreme Court precedent. (5) **Delaware Corporate Law Significance**: Delaware corporate precedents have national influence, so both decisions may affect corporate law development nationwide.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Delaware",
            "legal_domain": "corporate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 108,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a trademark infringement case and found that the Second Circuit and Ninth Circuit have conflicting precedents on the same legal issue. How does this circuit split affect my case in the Third Circuit?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Circuit split creates opportunity and uncertainty** for Third Circuit case. Circuit split analysis: (1) **No Controlling Precedent**: Neither Second nor Ninth Circuit precedent is controlling in Third Circuit. Both are persuasive authority only. (2) **Precedent Evaluation Factors**: (a) **Reasoning quality** - Which decision has better legal analysis?; (b) **Factual similarity** - Which case facts are more similar to yours?; (c) **Policy considerations** - Which approach better serves trademark law purposes?; (d) **Trend analysis** - Which view is gaining acceptance? (3) **Strategic Advantages**: (a) **Choice of precedent** - Argue for the circuit decision that favors your position; (b) **First impression** - Third Circuit has opportunity to choose the better-reasoned approach; (c) **Precedent creation** - Your case could establish Third Circuit precedent. (4) **Litigation Strategy**: (a) **Emphasize favorable circuit** - Argue its reasoning is superior; (b) **Distinguish unfavorable circuit** - Show factual or legal differences; (c) **Policy arguments** - Explain why your preferred approach serves trademark law better. (5) **Supreme Court Implications**: Circuit splits often lead to Supreme Court review, which could: (a) **Resolve conflict** - Establish national rule; (b) **Affect your case** - If cert petition filed while your case pending. (6) **Research Priority**: Examine district court decisions within Third Circuit and other circuits to identify emerging trends and strengthen your position.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 109,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm arguing a products liability case in California state court. There's a California Supreme Court case from 1963 that established strict liability, but manufacturing and technology have changed significantly. How do I analyze this precedent's current relevance?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **1963 California Supreme Court precedent remains controlling** but requires modern application analysis. Precedent durability assessment: (1) **Controlling Authority**: California Supreme Court precedent is binding on all California courts regardless of age, unless: (a) **Explicitly overruled** by California Supreme Court; (b) **Superseded by statute**; (c) **Constitutionally invalid**. (2) **Legal Principle vs. Application**: (a) **Core principle** - Strict liability doctrine likely remains valid; (b) **Factual application** - How it applies to modern technology/manufacturing may evolve; (c) **Policy rationale** - Consumer protection rationale may be stronger today. (3) **Evolutionary Analysis**: (a) **Subsequent precedents** - How have later California cases refined or expanded the doctrine?; (b) **Statutory changes** - Have California statutes modified products liability law?; (c) **Technological advancement** - How do new technologies fit within established framework? (4) **Strategic Application**: (a) **Emphasize core principle** - Strict liability doctrine established; (b) **Analogize to modern facts** - Show how principle applies to current technology; (c) **Policy arguments** - Argue consumer protection need is greater today. (5) **Potential Challenges**: (a) **Factual distinction** - Defendant may argue different technology requires different rule; (b) **Policy shifts** - Arguments that strict liability is outdated; (c) **Statutory preemption** - Federal regulations may affect state law. (6) **Research Strategy**: Find intermediate California appellate decisions applying 1963 precedent to modern cases, strengthening your argument for continued vitality.",
            "task_type": "precedent_analysis",
            "jurisdiction": "California",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 110,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a contract dispute in New York. The other party cited a 1950s New York Court of Appeals case, but I found a 2020 New York Court of Appeals case that seems to change the rule. How do I analyze this precedent evolution?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **2020 New York Court of Appeals case is controlling** and likely supersedes the 1950s precedent. Precedent evolution analysis: (1) **Hierarchical Authority**: Both cases are from New York's highest court, so both would normally be controlling. However, **later precedent typically supersedes earlier precedent** when they conflict. (2) **Precedent Relationship Analysis**: (a) **Explicit overruling** - Does 2020 case explicitly overrule 1950s case?; (b) **Implicit modification** - Does 2020 case change legal standard without explicitly overruling?; (c) **Factual distinction** - Are cases factually different enough to coexist?; (d) **Legal evolution** - Has contract law doctrine evolved between 1950s and 2020? (3) **Contemporary Application**: (a) **Modern contract principles** - 2020 case reflects current legal thinking; (b) **Commercial reality** - Recent case likely considers modern business practices; (c) **Judicial philosophy** - Contemporary approach to contract interpretation. (4) **Strategic Approach**: (a) **Emphasize 2020 precedent** as controlling current law; (b) **Distinguish 1950s case** - Show it's outdated or factually different; (c) **Legal evolution argument** - Explain how contract law has developed; (d) **Policy considerations** - Argue modern approach better serves commercial needs. (5) **Opponent's Counter-arguments**: (a) **Factual distinction** - 1950s case may still apply to different facts; (b) **Narrow holding** - 2020 case may have limited scope. (6) **Research Priority**: Find intermediate cases showing evolution from 1950s to 2020 to trace doctrinal development.",
            "task_type": "precedent_analysis",
            "jurisdiction": "New_York",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 111,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a federal criminal case and found a Supreme Court precedent that's directly on point, but there's also a recent circuit court decision that distinguishes it on facts very similar to mine. How do I evaluate the precedent's application?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Supreme Court precedent is controlling**, but circuit court distinction analysis is crucial. Precedent application framework: (1) **Hierarchical Authority**: Supreme Court precedent is controlling on all federal courts. Circuit court cannot overrule but can distinguish on facts or limit scope. (2) **Distinction Analysis**: (a) **Factual distinction** - Are factual differences legally significant?; (b) **Proper distinction** - Does circuit court correctly identify relevant factual differences?; (c) **Legal reasoning** - Is circuit court's legal analysis sound?; (d) **Scope limitation** - Does circuit court properly narrow Supreme Court precedent? (3) **Critical Evaluation**: (a) **Factual similarity** - How similar are your facts to Supreme Court case vs. circuit case?; (b) **Legal principle** - What is the core legal principle from Supreme Court?; (c) **Policy rationale** - What policy underlies Supreme Court decision? (4) **Strategic Arguments**: (a) **Direct application** - Argue Supreme Court precedent directly applies to your facts; (b) **Distinguish circuit case** - Show your facts are different from circuit case; (c) **Precedent fidelity** - Argue circuit court improperly narrowed Supreme Court precedent. (5) **Litigation Risks**: (a) **Circuit binding** - Circuit precedent is binding within that circuit; (b) **Appeal potential** - If circuit court incorrectly applied Supreme Court precedent; (c) **Cert petition** - Possible Supreme Court review if circuit created improper distinction. (6) **Research Strategy**: Find other circuits' interpretations of Supreme Court precedent to show proper application and support your position.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 112,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a constitutional law issue and found that the Supreme Court has granted cert in a case that could affect my pending litigation. How do I analyze this pending Supreme Court case's impact on my precedent analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Pending Supreme Court case creates strategic complexity** requiring careful timing and precedent preservation analysis. Cert grant implications: (1) **Current Controlling Precedent**: Until Supreme Court decides, existing precedent remains fully controlling. Don't wait for decision - proceed with current law. (2) **Potential Impact Assessment**: (a) **Scope of review** - What specific legal issues did Court grant cert on?; (b) **Precedent risk** - Could decision overrule or modify precedent favorable to your case?; (c) **Precedent opportunity** - Could decision strengthen your position?; (d) **Timing considerations** - When will decision likely be issued? (3) **Strategic Options**: (a) **Expedite proceedings** - Seek faster resolution before Supreme Court decides; (b) **Seek stay** - Request delay until Supreme Court clarifies law; (c) **Preserve arguments** - Ensure record preserves all legal theories; (d) **Hedge positions** - Argue under both current and potential future law. (4) **Litigation Timing**: (a) **Trial court strategy** - Argue under current precedent while noting potential changes; (b) **Appellate preservation** - Ensure appeals can benefit from new precedent; (c) **Settlement considerations** - Uncertainty may affect settlement dynamics. (5) **Precedent Research**: (a) **Oral argument analysis** - Listen for Court's concerns and likely outcome; (b) **Amicus briefs** - Understand various positions and arguments; (c) **Lower court decisions** - Analyze what Supreme Court is reviewing. (6) **Risk Management**: Prepare alternative legal theories in case primary precedent is overruled or modified.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 113,
            "difficulty": "advanced",
            "court_level": "federal_circuit",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a Texas family law case involving child custody. I found a Texas Supreme Court case from 2018 that supports my position, but the opposing party cited a 2022 Texas Court of Appeals case from a different district that disagrees. How do I analyze this precedent hierarchy?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Texas Supreme Court precedent (2018) is controlling** over the Court of Appeals decision. Texas hierarchy analysis: (1) **Controlling Authority**: Texas Supreme Court decisions are binding on all Texas courts, including all Courts of Appeals. The 2018 decision controls unless: (a) It's been overruled; (b) The appellate court properly distinguished it; (c) Statutory changes have modified the law. (2) **Court of Appeals Authority**: Texas Court of Appeals decisions are: (a) **Binding within district** - Control trial courts in that specific appellate district; (b) **Persuasive elsewhere** - Not binding on other appellate districts or trial courts outside the district; (c) **Subject to Supreme Court review** - Can be overruled by Texas Supreme Court. (3) **Precedent Conflict Analysis**: (a) **Improper disagreement** - Court of Appeals cannot simply disagree with Supreme Court; (b) **Factual distinction** - Appellate court may have distinguished on facts; (c) **Legal error** - Appellate court may have misapplied Supreme Court precedent. (4) **Strategic Response**: (a) **Emphasize Supreme Court precedent** as controlling statewide; (b) **Distinguish appellate case** - Show factual or legal differences; (c) **Jurisdictional limits** - Argue appellate case not binding in your district; (d) **Petition for review** - Appellate case may be reviewable if it conflicts with Supreme Court. (5) **Family Law Considerations**: (a) **Best interests standard** - Supreme Court precedent likely establishes framework; (b) **Factual application** - Appellate court may have applied framework to different facts. (6) **Research Strategy**: Find other appellate districts' decisions following Supreme Court precedent to show proper application.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Texas",
            "legal_domain": "family_law"
        },
        "extra_info": {
            "split": "train",
            "index": 114,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a securities fraud case in federal court. The SEC has published new guidance that seems to conflict with established circuit precedent. How do I analyze this administrative guidance versus judicial precedent?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Circuit precedent generally controls** over SEC guidance, but requires nuanced analysis of administrative law principles. Authority hierarchy analysis: (1) **Judicial vs. Administrative Authority**: (a) **Circuit precedent** - Binding interpretation of federal securities law; (b) **SEC guidance** - Administrative interpretation, not binding law but entitled to some deference; (c) **Conflict resolution** - Courts interpret law, agencies implement it. (2) **Chevron Deference Analysis**: (a) **Statutory ambiguity** - If statute is ambiguous, courts may defer to reasonable agency interpretation; (b) **Clear statutory language** - No deference if statute is clear; (c) **Guidance vs. regulation** - Formal regulations get more deference than informal guidance. (3) **Precedent Strength Assessment**: (a) **Circuit precedent age** - Older precedent may not reflect current legal thinking; (b) **Guidance authority** - Is it formal rulemaking or informal guidance?; (c) **Statutory basis** - What statutory provision underlies each position? (4) **Strategic Considerations**: (a) **Emphasize circuit precedent** - Argue binding judicial interpretation; (b) **Distinguish SEC guidance** - Show it's not binding or exceeds agency authority; (c) **Policy arguments** - Argue which interpretation better serves securities law purposes; (d) **Chevron analysis** - Argue statute is clear and precludes agency interpretation. (5) **Litigation Risks**: (a) **Agency enforcement** - SEC may enforce its interpretation regardless of circuit precedent; (b) **Appeal potential** - Circuit court may reconsider precedent in light of agency guidance; (c) **Supreme Court review** - Conflict between circuits and agencies may warrant high court review. (6) **Research Priority**: Analyze other circuits' treatment of similar SEC guidance to identify trends and strengthen position.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "securities_law"
        },
        "extra_info": {
            "split": "train",
            "index": 115,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a state constitutional issue in Pennsylvania. There's a U.S. Supreme Court case interpreting a similar federal constitutional provision, but also a Pennsylvania Supreme Court case that provides greater protection. Which precedent should I rely on?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Pennsylvania Supreme Court precedent controls** on state constitutional issues and can provide greater protection than federal minimums. Federalism analysis: (1) **Constitutional Floor Principle**: U.S. Supreme Court precedent establishes **minimum constitutional protection** that states cannot fall below, but states can provide greater protection under state constitutions. (2) **State Constitutional Independence**: (a) **Pennsylvania Supreme Court** - Final authority on Pennsylvania constitutional interpretation; (b) **Independent interpretation** - Can interpret state constitution more broadly than federal counterpart; (c) **Textual differences** - Pennsylvania constitution may have different language requiring different interpretation. (3) **Precedent Interaction**: (a) **Federal minimum** - Supreme Court precedent sets floor; (b) **State enhancement** - Pennsylvania precedent provides additional protection; (c) **No conflict** - Both can coexist with state providing greater rights. (4) **Strategic Advantage**: (a) **Dual constitutional basis** - Argue both federal and state constitutional violations; (b) **State-focused argument** - Emphasize Pennsylvania constitutional protections; (c) **Precedent strength** - Pennsylvania Supreme Court precedent is controlling on state issues. (5) **Litigation Benefits**: (a) **Greater protection** - State constitution may provide broader rights; (b) **Jurisdictional advantage** - State court expertise in state constitutional law; (c) **Appeal limitations** - U.S. Supreme Court review of state constitutional grounds is limited. (6) **Research Strategy**: (a) **Textual comparison** - Compare Pennsylvania and federal constitutional language; (b) **Historical analysis** - Understand Pennsylvania constitutional development; (c) **Policy considerations** - Argue state-specific reasons for greater protection.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Pennsylvania",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 116,
            "difficulty": "advanced",
            "court_level": "state_supreme",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a breach of contract case in Michigan. The other party cited a very old Michigan Supreme Court case from 1923, but I found several recent Michigan Court of Appeals cases that seem to take a different approach. How do I analyze this precedent evolution?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **1923 Michigan Supreme Court case remains controlling** unless explicitly overruled, but recent Court of Appeals cases may show evolving interpretation. Precedent hierarchy analysis: (1) **Hierarchical Authority**: Michigan Supreme Court precedent (1923) is controlling over all Michigan courts, including Court of Appeals. Age alone doesn't diminish binding authority. (2) **Precedent Vitality Assessment**: (a) **Explicit overruling** - Has Michigan Supreme Court explicitly overruled 1923 case?; (b) **Implicit modification** - Have subsequent Supreme Court cases modified the principle?; (c) **Statutory supersession** - Has Michigan legislature changed relevant contract law? (3) **Court of Appeals Evolution**: (a) **Proper application** - Are appellate cases properly applying 1923 precedent to modern facts?; (b) **Factual distinction** - Are cases distinguishable on facts?; (c) **Legal development** - Are appellate courts showing evolution in contract law thinking?; (d) **Supreme Court review** - Have any appellate decisions been reviewed by Supreme Court? (4) **Strategic Approach**: (a) **Emphasize controlling precedent** - 1923 case is binding law; (b) **Distinguish appellate cases** - Show factual or legal differences; (c) **Evolution argument** - Argue contract law has developed since 1923; (d) **Policy considerations** - Modern commercial practices may require different approach. (5) **Litigation Risks**: (a) **Appellate trends** - Multiple recent cases suggest possible Supreme Court review; (b) **Appeal potential** - Issue may be ripe for Supreme Court clarification. (6) **Research Strategy**: Look for Michigan Supreme Court cases since 1923 addressing similar contract issues to trace doctrinal development.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Michigan",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 117,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a federal environmental law case. There's a Supreme Court case from 2015 that established a test, but EPA has issued new regulations that seem to interpret the test differently. How do I analyze this precedent versus agency interpretation?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Supreme Court precedent (2015) is controlling**, but EPA regulations may warrant deference under administrative law principles. Administrative law analysis: (1) **Hierarchical Authority**: Supreme Court precedent is supreme law of the land. EPA regulations cannot override Supreme Court interpretation but may fill gaps or provide implementation details. (2) **Chevron Analysis Framework**: (a) **Step 1** - Is Supreme Court precedent clear and unambiguous?; (b) **Step 2** - If ambiguous, is EPA interpretation reasonable?; (c) **Deference scope** - Deference applies only to statutory interpretation, not constitutional issues. (3) **Precedent-Regulation Relationship**: (a) **Consistent implementation** - EPA regulations may provide detailed implementation of Supreme Court test; (b) **Impermissible contradiction** - EPA cannot contradict clear Supreme Court holding; (c) **Gap-filling** - EPA may interpret areas not addressed by Supreme Court. (4) **Strategic Analysis**: (a) **Emphasize Supreme Court precedent** - Argue it clearly establishes binding test; (b) **Challenge EPA interpretation** - Show it contradicts or exceeds Court's holding; (c) **Alternative interpretation** - Argue EPA should interpret regulations consistently with precedent. (5) **Litigation Considerations**: (a) **Deference limits** - Courts don't defer to agency interpretations that contradict Supreme Court; (b) **Reasonableness standard** - EPA interpretation must be reasonable even if entitled to deference; (c) **Policy arguments** - Environmental protection goals vs. legal clarity. (6) **Research Priority**: (a) **Circuit treatment** - How have other circuits addressed this precedent-regulation tension?; (b) **Legislative history** - What did Congress intend regarding EPA authority?; (c) **Statutory construction** - How do statute, precedent, and regulation interact?",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "environmental_law"
        },
        "extra_info": {
            "split": "train",
            "index": 118,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a criminal case in Virginia state court. There's a Fourth Circuit case that seems to support the defendant's position, but also a Virginia Supreme Court case that supports the prosecution. The legal issue involves both federal constitutional law and Virginia criminal procedure. How do I analyze these precedents?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Dual precedent analysis required** - Fourth Circuit controls federal constitutional issues, Virginia Supreme Court controls state procedure. Legal domain separation: (1) **Federal Constitutional Issues**: (a) **Fourth Circuit precedent** - Binding on federal constitutional criminal procedure; (b) **Supreme Court floor** - Establishes minimum constitutional protections; (c) **State cannot go below** - Virginia cannot provide less protection than federal minimum. (2) **State Criminal Procedure**: (a) **Virginia Supreme Court precedent** - Controlling on Virginia criminal procedure rules; (b) **Independent state grounds** - State can provide greater protection than federal minimum; (c) **Procedural autonomy** - States control their own criminal procedure within constitutional limits. (3) **Precedent Interaction Analysis**: (a) **Constitutional floor** - Does Fourth Circuit precedent establish minimum protection?; (b) **State enhancement** - Does Virginia precedent provide additional protection?; (c) **Actual conflict** - Do precedents truly conflict or address different issues? (4) **Strategic Approach**: (a) **Identify specific issues** - Separate federal constitutional from state procedural issues; (b) **Apply appropriate precedent** - Fourth Circuit for federal, Virginia Supreme Court for state; (c) **Dual arguments** - Argue both federal and state grounds. (5) **Litigation Advantages**: (a) **Multiple grounds** - Win on either federal or state basis; (b) **Appellate protection** - State constitutional grounds limit U.S. Supreme Court review; (c) **Forum choice** - State court may be more receptive to state constitutional arguments. (6) **Research Strategy**: (a) **Issue separation** - Clearly identify which issues are federal vs. state; (b) **Precedent harmony** - Show how both precedents can coexist; (c) **Policy arguments** - Argue for greater state protection if applicable.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Virginia",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 119,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a personal injury case in Ohio. I found a case from the Ohio Supreme Court that establishes a favorable standard, but it was decided by a 4-3 vote with strong dissents. How does the narrow majority affect the precedent's strength?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Ohio Supreme Court precedent is fully controlling** despite narrow majority, but dissenting opinions may signal future instability. Precedent strength analysis: (1) **Controlling Authority**: Ohio Supreme Court decisions are binding on all Ohio courts regardless of vote margin. A 4-3 majority creates the same binding precedent as a unanimous decision. (2) **Precedent Stability Factors**: (a) **Narrow majority** - May indicate divided court and potential for future reconsideration; (b) **Strong dissents** - May provide roadmap for future challenges; (c) **Judicial turnover** - New justices might view issue differently; (d) **Doctrinal tensions** - Division may reflect broader legal uncertainty. (3) **Dissent Analysis**: (a) **Legal reasoning** - Do dissents identify weaknesses in majority opinion?; (b) **Policy arguments** - Do dissents offer compelling alternative approaches?; (c) **Precedent criticism** - Do dissents suggest majority misapplied existing law?; (d) **Future influence** - Dissents may become majority view if court composition changes. (4) **Strategic Considerations**: (a) **Emphasize majority** - It's controlling law regardless of margin; (b) **Distinguish dissents** - Show why majority reasoning is superior; (c) **Stability arguments** - Argue for precedent stability and predictability; (d) **Policy support** - Reinforce majority's policy reasoning. (5) **Litigation Risks**: (a) **Future overruling** - Narrow precedents more likely to be reconsidered; (b) **Factual distinction** - Opposing parties may try to distinguish narrow precedent; (c) **Appeal strategy** - Consider whether to seek Supreme Court review while precedent is favorable. (6) **Research Priority**: Monitor Ohio Supreme Court composition and subsequent cases to assess precedent stability.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Ohio",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 120,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a First Amendment case in federal court. There's a Supreme Court case from 1969 that's directly on point, but I also found a very recent district court case from last month that applied it to similar facts and reached the opposite conclusion. How do I analyze this apparent contradiction?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Supreme Court precedent (1969) is controlling** and the district court may have misapplied it. Precedent application analysis: (1) **Controlling Authority**: Supreme Court precedent is binding on all federal courts. District court decisions cannot override Supreme Court precedent, only apply it to specific facts. (2) **Contradiction Analysis**: (a) **Factual distinction** - Are the cases factually distinguishable in legally significant ways?; (b) **Legal misapplication** - Did district court misunderstand or misapply Supreme Court precedent?; (c) **Precedent evolution** - Has intervening Supreme Court authority modified the 1969 precedent?; (d) **Constitutional interpretation** - Are there different ways to interpret the constitutional principle? (3) **Precedent Vitality Check**: (a) **Subsequent Supreme Court cases** - Has the Court refined or limited the 1969 precedent?; (b) **Circuit treatment** - How have circuit courts applied the 1969 precedent?; (c) **Doctrinal development** - Has First Amendment doctrine evolved since 1969? (4) **Strategic Response**: (a) **Emphasize Supreme Court precedent** - It's controlling law; (b) **Distinguish district court case** - Show factual or legal differences; (c) **Argue misapplication** - Demonstrate district court incorrectly applied precedent; (d) **Policy consistency** - Show your interpretation serves First Amendment values. (5) **Litigation Considerations**: (a) **Precedent strength** - 1969 precedent may be well-established but consider evolution; (b) **Circuit appeal** - District court error may be grounds for appeal; (c) **Supreme Court review** - If circuit courts split, may warrant high court review. (6) **Research Strategy**: (a) **Circuit survey** - How do other circuits apply 1969 precedent?; (b) **Factual analysis** - Identify key factual differences; (c) **Doctrinal mapping** - Trace First Amendment development since 1969.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 121,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a corporate law case in Delaware. There's a Delaware Supreme Court case that's clearly controlling, but I also found a well-reasoned law review article by a prominent legal scholar that strongly criticizes the precedent. How do I evaluate this scholarly criticism?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Delaware Supreme Court precedent remains fully controlling** regardless of scholarly criticism, but academic analysis may provide strategic insights. Precedent versus scholarship analysis: (1) **Hierarchical Authority**: Delaware Supreme Court precedent is binding law. Legal scholarship, however prestigious, cannot override or diminish controlling precedent. (2) **Scholarly Criticism Value**: (a) **Analytical insight** - May identify weaknesses in precedent reasoning; (b) **Policy arguments** - May provide compelling policy reasons for different approach; (c) **Doctrinal context** - May place precedent in broader legal framework; (d) **Future influence** - May influence future judicial decisions or legislative action. (3) **Strategic Considerations**: (a) **Follow controlling precedent** - Must apply Delaware Supreme Court precedent; (b) **Distinguish if possible** - Use scholarly criticism to distinguish case on facts; (c) **Policy arguments** - Incorporate scholar's policy analysis where appropriate; (d) **Appeal strategy** - Scholarly criticism may support arguments for precedent reconsideration. (4) **Litigation Applications**: (a) **Factual distinction** - Use criticism to argue precedent doesn't apply to your facts; (b) **Narrow interpretation** - Argue for limited reading of precedent; (c) **Legislative advocacy** - Scholarly criticism may support legislative reform; (d) **Future cases** - Criticism may influence future judicial decisions. (5) **Delaware Corporate Law Significance**: (a) **National influence** - Delaware precedents affect corporate law nationwide; (b) **Judicial attention** - Delaware Supreme Court may be aware of scholarly criticism; (c) **Business impact** - Scholarly criticism may reflect business community concerns. (6) **Research Strategy**: (a) **Additional scholarship** - Find supporting academic authority; (b) **Judicial citations** - See if other courts have noted scholarly criticism; (c) **Legislative response** - Check if criticism has prompted legislative consideration.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Delaware",
            "legal_domain": "corporate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 122,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a federal employment discrimination case. There's a Supreme Court case from 2020 that established a new standard, but I found several circuit court decisions from before 2020 that used a different standard. How do I analyze the precedent shift?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **2020 Supreme Court precedent is controlling** and supersedes all prior circuit court decisions on the same issue. Precedent transition analysis: (1) **Controlling Authority**: Supreme Court precedent (2020) is binding on all federal courts and overrides contrary circuit court decisions, even if those decisions were correct under prior law. (2) **Precedent Supersession**: (a) **Explicit overruling** - Did Supreme Court explicitly overrule prior circuit decisions?; (b) **Standard change** - Does new standard fundamentally alter legal analysis?; (c) **Retroactive application** - Does new standard apply to cases pending on appeal?; (d) **Transitional issues** - How do courts handle cases decided under old standard? (3) **Pre-2020 Precedent Status**: (a) **No longer controlling** - Circuit decisions using old standard are not binding; (b) **Historical context** - May provide insight into legal development; (c) **Factual applications** - May show how legal standards apply to specific facts; (d) **Policy considerations** - May explain rationale for different approaches. (4) **Strategic Application**: (a) **Emphasize 2020 precedent** - It's current controlling law; (b) **Distinguish old cases** - Show they used superseded standard; (c) **Apply new standard** - Demonstrate how 2020 precedent applies to your facts; (d) **Policy arguments** - Explain why new standard better serves employment law. (5) **Litigation Considerations**: (a) **Retroactivity** - New standard may apply to pending cases; (b) **Appellate strategy** - Cases decided under old standard may be subject to review; (c) **Settlement impact** - Legal certainty may affect settlement negotiations. (6) **Research Priority**: (a) **Post-2020 applications** - How have courts applied new standard?; (b) **Circuit implementation** - How have circuits adapted to new precedent?; (c) **Practical implications** - How does new standard affect employment practices?",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 123,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a tort case in Georgia state court. There's a Georgia Supreme Court case that's on point, but I also found a Georgia Court of Appeals case from the same year that seems to interpret the Supreme Court case differently. How do I analyze this same-year precedent interpretation?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Georgia Supreme Court precedent is controlling** and the Court of Appeals case may have misinterpreted it. Same-year precedent analysis: (1) **Hierarchical Authority**: Georgia Supreme Court precedent is binding on all Georgia courts, including the Court of Appeals. Same-year timing doesn't change this hierarchy. (2) **Interpretation Analysis**: (a) **Correct application** - Did Court of Appeals properly interpret Supreme Court precedent?; (b) **Factual distinction** - Are the cases factually different enough to warrant different outcomes?; (c) **Scope misunderstanding** - Did Court of Appeals misunderstand the scope of Supreme Court holding?; (d) **Timing factors** - Did Court of Appeals have access to complete Supreme Court opinion? (3) **Precedent Relationship**: (a) **Direct application** - Is Court of Appeals case a direct application of Supreme Court precedent?; (b) **Distinguishable facts** - Are factual differences legally significant?; (c) **Legal principle** - What is the core legal principle from Supreme Court?; (d) **Policy consistency** - Are both decisions consistent with underlying policy? (4) **Strategic Approach**: (a) **Emphasize Supreme Court precedent** - It's the controlling authority; (b) **Distinguish Court of Appeals case** - Show factual or legal differences; (c) **Argue misinterpretation** - Demonstrate Court of Appeals misapplied Supreme Court precedent; (d) **Direct application** - Show how Supreme Court precedent directly applies to your case. (5) **Litigation Risks**: (a) **Appellate review** - Court of Appeals interpretation may be subject to Supreme Court review; (b) **Precedent confusion** - Different interpretations may create uncertainty; (c) **Trial court guidance** - Judge may need guidance on which interpretation to follow. (6) **Research Strategy**: (a) **Subsequent cases** - How have other courts interpreted the Supreme Court precedent?; (b) **Factual analysis** - Identify key factual differences between cases; (c) **Legal development** - Has Supreme Court clarified its precedent since both decisions?",
            "task_type": "precedent_analysis",
            "jurisdiction": "Georgia",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 124,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a contract case in Illinois. There's a recent Illinois Supreme Court case that's directly on point, but I also found several cases from other states that have rejected the Illinois approach. How do I analyze this interstate precedent conflict?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Illinois Supreme Court precedent is controlling** within Illinois, but interstate conflict may signal broader legal issues. Interstate precedent analysis: (1) **Controlling Authority**: Illinois Supreme Court precedent is binding on all Illinois courts. Other states' decisions are not binding but may be persuasive or instructive. (2) **Interstate Conflict Significance**: (a) **Legal development** - May indicate evolving area of contract law; (b) **Policy debate** - Different states may have different policy preferences; (c) **Uniform law considerations** - May affect interstate commerce and legal predictability; (d) **Supreme Court interest** - Interstate splits may warrant U.S. Supreme Court review if federal issues involved. (3) **Other States' Precedent Value**: (a) **Persuasive authority** - Well-reasoned decisions may influence future Illinois decisions; (b) **Policy arguments** - May provide compelling reasons for different approach; (c) **Practical consequences** - May show practical problems with Illinois approach; (d) **Comparative analysis** - May help evaluate wisdom of Illinois rule. (4) **Strategic Considerations**: (a) **Emphasize Illinois precedent** - It's controlling law in Illinois; (b) **Distinguish other states** - Show factual or legal differences; (c) **Policy arguments** - Explain why Illinois approach is superior; (d) **Interstate commerce** - Address any interstate commerce concerns. (5) **Litigation Implications**: (a) **Controlling precedent** - Must follow Illinois Supreme Court regardless of other states; (b) **Future development** - Illinois court might reconsider if strong contrary trend develops; (c) **Legislative attention** - Interstate conflict might prompt legislative action; (d) **Business impact** - May affect contracts with interstate implications. (6) **Research Strategy**: (a) **Trend analysis** - Are more states adopting the contrary approach?; (b) **Academic commentary** - What do scholars say about different approaches?; (c) **Practical consequences** - How do different rules affect business and legal practice?",
            "task_type": "precedent_analysis",
            "jurisdiction": "Illinois",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 125,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a criminal appeal in the Ninth Circuit. There's a Supreme Court case that established a constitutional rule, but the Ninth Circuit has a line of cases that has consistently interpreted it more narrowly than other circuits. How do I analyze this circuit-specific interpretation?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Ninth Circuit precedent is controlling** within the circuit, but narrow interpretation may create circuit split requiring analysis. Circuit-specific precedent analysis: (1) **Hierarchical Authority**: Supreme Court precedent establishes constitutional rule, but Ninth Circuit precedent controls how that rule is applied within the circuit. (2) **Interpretation Legitimacy**: (a) **Proper application** - Is Ninth Circuit properly applying Supreme Court precedent to specific facts?; (b) **Narrow interpretation** - Are constitutional minimums being met while applying rule narrowly?; (c) **Circuit precedent** - Are Ninth Circuit cases internally consistent?; (d) **Supreme Court constraints** - Is interpretation within bounds of Supreme Court precedent? (3) **Circuit Split Analysis**: (a) **Intercircuit conflict** - Do other circuits interpret Supreme Court precedent more broadly?; (b) **Majority position** - Are most circuits taking broader interpretation?; (c) **Factual distinctions** - Are circuit differences based on different factual patterns?; (d) **Legal reasoning** - Which interpretation better serves constitutional purposes? (4) **Strategic Approach**: (a) **Follow circuit precedent** - It's controlling within Ninth Circuit; (b) **Distinguish other circuits** - Show factual or legal differences; (c) **Argue proper interpretation** - Demonstrate Ninth Circuit interpretation is correct; (d) **Policy arguments** - Explain why narrow interpretation serves constitutional values. (5) **Litigation Considerations**: (a) **Binding precedent** - Must follow Ninth Circuit interpretation within circuit; (b) **En banc review** - Circuit split may warrant en banc consideration; (c) **Supreme Court review** - Circuit split may interest Supreme Court; (d) **Precedent strength** - Consistent line of cases suggests settled circuit law. (6) **Research Strategy**: (a) **Circuit survey** - Document extent of circuit split; (b) **Supreme Court signals** - Any indication of Supreme Court views on proper interpretation?; (c) **Academic commentary** - Scholarly analysis of different circuit approaches.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 126,
            "difficulty": "advanced",
            "court_level": "federal_circuit",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm in a products liability case in California. There's a California Supreme Court case that's favorable to my position, but the California legislature just passed a statute that seems to modify the common law rule. How do I analyze this precedent versus statutory change?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **New California statute likely supersedes** common law precedent, but requires careful statutory interpretation analysis. Precedent versus statute analysis: (1) **Legal Hierarchy**: Statutes generally supersede common law precedents when they conflict. California Supreme Court precedent established common law rule, but legislature can change it through statute. (2) **Statutory Interpretation**: (a) **Express override** - Does statute explicitly override common law rule?; (b) **Implicit modification** - Does statute impliedly change common law without explicit statement?; (c) **Scope of change** - What aspects of common law does statute modify?; (d) **Effective date** - When does statute take effect and does it apply to your case? (3) **Precedent Survival Analysis**: (a) **Partial supersession** - Does statute modify only part of common law rule?; (b) **Complementary law** - Can statute and common law coexist?; (c) **Interpretive guidance** - Does precedent help interpret ambiguous statutory language?; (d) **Policy continuity** - Are statute and precedent consistent in policy goals? (4) **Strategic Approach**: (a) **Statutory interpretation** - Argue statute should be interpreted consistently with favorable precedent; (b) **Scope limitation** - Argue statute has limited scope that doesn't affect your case; (c) **Effective date** - Argue statute doesn't apply to your case due to timing; (d) **Constitutional challenges** - Consider whether statute violates constitutional rights. (5) **Litigation Considerations**: (a) **Transitional period** - Courts may need guidance on statute-precedent interaction; (b) **Appellate review** - Statutory interpretation questions may warrant appellate review; (c) **Legislative intent** - Review legislative history for interpretation guidance; (d) **Retroactivity** - Does statute apply to pending cases? (6) **Research Strategy**: (a) **Legislative history** - Understand legislature's intent regarding common law; (b) **Statutory construction** - Apply rules of statutory interpretation; (c) **Similar statutes** - How have courts interpreted similar legislative changes?",
            "task_type": "precedent_analysis",
            "jurisdiction": "California",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 127,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching a First Amendment case and found a Supreme Court precedent that's directly on point. However, I also found a concurring opinion in that same case that takes a different analytical approach. How do I analyze the relationship between the majority opinion and the concurrence?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Majority opinion is controlling precedent**; concurring opinion provides persuasive authority and may signal future doctrinal development. Majority versus concurrence analysis: (1) **Precedential Authority**: Only the majority opinion creates binding precedent. Concurring opinions are persuasive authority that may influence future decisions but don't establish controlling law. (2) **Precedent Scope**: (a) **Holding** - What is the specific legal rule established by majority?; (b) **Reasoning** - What analytical framework does majority use?; (c) **Factual application** - How does majority apply law to facts?; (d) **Future guidance** - What does majority suggest for future cases? (3) **Concurrence Analysis**: (a) **Doctrinal alternative** - Does concurrence propose different legal framework?; (b) **Narrower grounds** - Does concurrence reach same result through narrower reasoning?; (c) **Future direction** - Does concurrence suggest evolution of doctrine?; (d) **Policy considerations** - Does concurrence emphasize different policy values? (4) **Strategic Considerations**: (a) **Emphasize majority** - It's controlling precedent; (b) **Invoke concurrence** - Use as persuasive authority for broader or narrower interpretation; (c) **Doctrinal evolution** - Argue concurrence represents better approach for future development; (d) **Policy arguments** - Use concurrence's policy analysis to support position. (5) **Litigation Applications**: (a) **Primary reliance** - Majority opinion is primary precedent; (b) **Supplementary authority** - Concurrence may provide additional support; (c) **Factual distinction** - Concurrence may offer different analytical framework; (d) **Appeal strategy** - Concurrence may influence higher court thinking. (6) **Research Strategy**: (a) **Subsequent treatment** - How have later cases treated majority vs. concurrence?; (b) **Circuit application** - Do circuits follow majority or show concurrence influence?; (c) **Academic commentary** - Do scholars favor majority or concurrence approach?",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 128,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm working on a commercial litigation case in New York. There's a New York Court of Appeals case that's directly on point, but I also found a federal district court case in the Southern District of New York that applied New York law differently. How do I analyze this federal versus state court interpretation?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **New York Court of Appeals precedent is controlling** on New York state law issues; federal district court interpretation is not binding. Federal-state precedent analysis: (1) **State Law Authority**: New York Court of Appeals is the final authority on New York state law. Federal courts applying state law must follow state supreme court precedent. (2) **Federal Court Role**: (a) **Erie doctrine** - Federal courts must apply state law as interpreted by state supreme court; (b) **Predictive interpretation** - Federal courts predict how state supreme court would rule on novel issues; (c) **No independent authority** - Federal courts cannot create state law precedent; (d) **Binding effect** - Federal interpretations don't bind state courts. (3) **District Court Analysis**: (a) **Misapplication** - Did federal court misapply New York Court of Appeals precedent?; (b) **Factual distinction** - Are cases factually different enough to warrant different results?; (c) **Novel issue** - Was federal court addressing issue not covered by state precedent?; (d) **Predictive error** - Did federal court incorrectly predict state law development? (4) **Strategic Approach**: (a) **Emphasize state precedent** - It's controlling on state law issues; (b) **Distinguish federal case** - Show factual or legal differences; (c) **Argue misapplication** - Demonstrate federal court misapplied state law; (d) **State court forum** - Consider whether state court is preferable forum. (5) **Litigation Considerations**: (a) **Forum choice** - State courts may be more reliable for state law interpretation; (b) **Precedent value** - Federal district court decisions have no precedential value for state law; (c) **Appellate review** - Federal court errors may be corrected on appeal; (d) **Certification** - Federal courts may certify difficult state law questions to state supreme court. (6) **Research Strategy**: (a) **State law development** - Focus on New York state court decisions; (b) **Federal treatment** - How have other federal courts applied New York precedent?; (c) **Certification history** - Has this issue been certified to New York Court of Appeals?",
            "task_type": "precedent_analysis",
            "jurisdiction": "New_York",
            "legal_domain": "commercial_law"
        },
        "extra_info": {
            "split": "train",
            "index": 129,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm researching an employment law case in federal court. There's a Supreme Court case that established a framework, but I found a recent EEOC guidance document that interprets the framework differently than several circuit courts. How do I analyze this administrative guidance versus judicial precedent conflict?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis: **Circuit court precedent generally controls** over EEOC guidance, but administrative interpretation may warrant some deference. Administrative versus judicial authority analysis: (1) **Precedent Hierarchy**: Supreme Court precedent establishes framework, circuit courts interpret and apply it, EEOC guidance provides administrative interpretation. Courts interpret law; agencies implement it. (2) **Deference Analysis**: (a) **Chevron deference** - May apply if Supreme Court precedent leaves ambiguity for agency interpretation; (b) **Skidmore deference** - EEOC guidance may receive respect based on thoroughness and consistency; (c) **Guidance limitations** - Informal guidance receives less deference than formal regulations; (d) **Judicial interpretation** - Courts have primary authority to interpret employment statutes. (3) **Authority Conflict**: (a) **Circuit precedent** - Binding interpretation within circuit; (b) **EEOC guidance** - Administrative view but not binding law; (c) **Conflict resolution** - Courts generally favor judicial interpretation over agency guidance; (d) **Statutory interpretation** - Focus on what statute requires, not agency preference. (4) **Strategic Approach**: (a) **Emphasize circuit precedent** - It's binding judicial interpretation; (b) **Distinguish EEOC guidance** - Show it exceeds agency authority or contradicts precedent; (c) **Statutory construction** - Argue courts should interpret statute independently; (d) **Policy arguments** - Explain why judicial interpretation better serves employment law. (5) **Litigation Considerations**: (a) **Agency enforcement** - EEOC may enforce its interpretation regardless of circuit precedent; (b) **Intercircuit conflict** - May warrant Supreme Court review; (c) **Regulatory authority** - Distinguish between formal regulations and informal guidance; (d) **Business certainty** - Conflicting interpretations create compliance uncertainty. (6) **Research Strategy**: (a) **Circuit survey** - How do other circuits treat EEOC guidance?; (b) **Deference analysis** - What level of deference is appropriate?; (c) **Statutory construction** - Focus on statutory text and Supreme Court framework.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 130,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": [],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Johnson v. Tech Corp, a employment discrimination case. The court cited three precedents: McDonnell Douglas Corp. v. Green (1973), Reeves v. Sanderson Plumbing (2000), and Desert Palace v. Costa (2003). Can you explain how each precedent supports the case and why they were cited?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Johnson v. Tech Corp Employment Discrimination Case: **McDonnell Douglas Corp. v. Green (1973)**: *How it helps*: Establishes the foundational burden-shifting framework for employment discrimination cases. Provides the three-step analysis: (1) plaintiff establishes prima facie case; (2) defendant articulates legitimate, non-discriminatory reason; (3) plaintiff proves pretext. *Why cited*: **Controlling precedent** - Supreme Court decision binding on all courts for discrimination analysis. Essential foundation for any employment discrimination claim. **Reeves v. Sanderson Plumbing (2000)**: *How it helps*: Clarifies that plaintiff can satisfy pretext burden by showing defendant's stated reason is false, without additional evidence of discriminatory intent. Allows jury to infer discrimination from disbelief of employer's explanation plus prima facie case. *Why cited*: **Precedent refinement** - Modifies McDonnell Douglas framework by lowering plaintiff's burden at pretext stage. Critical for cases where plaintiff has strong evidence of pretext but limited direct evidence of bias. **Desert Palace v. Costa (2003)**: *How it helps*: Establishes that direct evidence is not required for mixed-motive discrimination claims under Title VII. Plaintiff can prove discrimination through circumstantial evidence alone. Expands available theories of liability. *Why cited*: **Evidentiary standard** - Allows discrimination claims to proceed with circumstantial evidence. Essential when plaintiff lacks smoking gun evidence but has strong circumstantial case. **Synergistic Effect**: These three precedents create comprehensive framework: McDonnell Douglas provides structure, Reeves eases pretext burden, Desert Palace expands evidence types. Together they maximize plaintiff's chances while maintaining legal rigor.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 131,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["McDonnell Douglas Corp. v. Green (1973)", "Reeves v. Sanderson Plumbing (2000)", "Desert Palace v. Costa (2003)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Smith v. ABC Hospital, a medical malpractice case in California. The court cited Landeros v. Flood (1976), Helling v. Carey (1974), and Tarasoff v. Regents of University of California (1976). How does each precedent support the case?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Smith v. ABC Hospital Medical Malpractice Case: **Landeros v. Flood (1976)**: *How it helps*: California Supreme Court case establishing that physicians have duty to use reasonable care in diagnosis and treatment. Creates framework for determining when failure to diagnose constitutes negligence. Establishes that obvious symptoms requiring investigation create liability when ignored. *Why cited*: **State controlling precedent** - Binding California authority on medical malpractice standards. Directly addresses diagnostic negligence which appears central to Smith case. **Helling v. Carey (1974)**: *How it helps*: Washington case holding that compliance with customary medical practice doesn't automatically shield from liability. Court can find negligence even when physician followed profession's standard practices if those practices are inadequate. *Why cited*: **Persuasive authority** - Though not binding in California, provides compelling reasoning that professional customs aren't dispositive. Strengthens argument that defendant's conduct was negligent despite following some medical protocols. **Tarasoff v. Regents of University of California (1976)**: *How it helps*: California Supreme Court case establishing heightened duty of care when patient safety is involved. Creates precedent for expanded physician duties beyond traditional doctor-patient relationship. Emphasizes protective obligations when harm is foreseeable. *Why cited*: **Expanding duty concept** - California precedent showing courts will expand medical duties when patient protection requires it. Demonstrates California's progressive approach to medical liability. **Strategic Precedent Use**: Landeros provides core malpractice framework, Helling undermines 'industry standard' defense, Tarasoff shows California's willingness to expand medical duties. Together they create multi-layered attack on defendant's conduct while establishing California's patient-protective approach.",
            "task_type": "precedent_analysis",
            "jurisdiction": "California",
            "legal_domain": "medical_malpractice_law"
        },
        "extra_info": {
            "split": "train",
            "index": 132,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Landeros v. Flood (1976)", "Helling v. Carey (1974)", "Tarasoff v. Regents of University of California (1976)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Williams v. State, a criminal appeal involving search and seizure. The court cited Terry v. Ohio (1968), United States v. Mendenhall (1980), and Florida v. Bostick (1991). Explain how each precedent supports the case."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Williams v. State Criminal Search and Seizure Appeal: **Terry v. Ohio (1968)**: *How it helps*: Supreme Court case establishing 'Terry stop' framework - police can conduct investigatory stops with reasonable suspicion of criminal activity. Creates standard for evaluating whether police encounter constituted seizure and whether it was justified. Provides balancing test between public safety and individual liberty. *Why cited*: **Foundational precedent** - Supreme Court authority defining constitutional standards for police stops. Essential for determining if initial police contact with Williams was lawful. **United States v. Mendenhall (1980)**: *How it helps*: Supreme Court case establishing test for when police encounter becomes 'seizure' requiring constitutional justification. Creates objective standard: would reasonable person feel free to leave? Provides framework for distinguishing voluntary encounters from seizures. *Why cited*: **Seizure determination** - Critical for determining whether Williams was 'seized' requiring justification under Fourth Amendment. If no seizure occurred, evidence is admissible regardless of officer's subjective intent. **Florida v. Bostick (1991)**: *How it helps*: Supreme Court case refining Mendenhall test for determining seizures. Establishes that inherently coercive environment doesn't automatically create seizure - focus remains on whether reasonable person would feel free to leave or refuse cooperation. *Why cited*: **Modern application** - Updates seizure analysis for various police encounter contexts. Provides most current framework for evaluating when police conduct crosses line into seizure territory. **Precedent Progression**: Terry establishes basic stop framework, Mendenhall creates seizure test, Bostick refines application. Together they provide complete analytical framework: (1) Was Williams seized? (Mendenhall/Bostick) (2) If yes, was seizure justified? (Terry). Sequential analysis maximizes chances of suppressing evidence if any constitutional violation occurred.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "criminal_law"
        },
        "extra_info": {
            "split": "train",
            "index": 133,
            "difficulty": "intermediate",
            "court_level": "state_supreme",
            "case_citations": ["Terry v. Ohio (1968)", "United States v. Mendenhall (1980)", "Florida v. Bostick (1991)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Brown v. Manufacturing Co., a products liability case in Texas. The court cited Greenman v. Yuba Power Products (1963), Barker v. Lull Engineering (1978), and Cronin v. J.B.E. Olson Corp. (1972). How does each precedent help the case?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Brown v. Manufacturing Co. Products Liability Case: **Greenman v. Yuba Power Products (1963)**: *How it helps*: California Supreme Court case establishing strict liability theory in products liability. Eliminates need to prove manufacturer negligence - liability attaches if product was defective and unreasonably dangerous. Revolutionary shift from negligence to strict liability doctrine. *Why cited*: **Doctrine foundation** - Though California precedent, Greenman established strict liability theory adopted nationwide. Provides conceptual foundation for products liability without requiring negligence proof. **Cronin v. J.B.E. Olson Corp. (1972)**: *How it helps*: California case eliminating 'unreasonably dangerous' requirement from strict liability analysis. Focuses liability on whether product failed to perform as safely as ordinary consumer would expect. Simplifies plaintiff's burden by removing confusing 'unreasonably dangerous' standard. *Why cited*: **Burden reduction** - Shows evolution toward more plaintiff-friendly strict liability standards. Demonstrates trend away from complex liability requirements toward consumer expectation focus. **Barker v. Lull Engineering (1978)**: *How it helps*: California case establishing dual test for design defects: (1) consumer expectation test, or (2) risk-benefit analysis. Provides alternative theories when consumer expectation test fails. Shifts burden to defendant on risk-benefit analysis if plaintiff makes preliminary showing. *Why cited*: **Modern framework** - Provides current analytical structure for design defect claims. Offers multiple paths to liability and shifts burden to defendant, maximizing plaintiff's chances. **Strategic Citation Pattern**: Greenman establishes strict liability foundation, Cronin removes barriers to liability, Barker provides modern dual-test framework. Though all California cases, they represent evolution of products liability law adopted in many states. Texas court citing them suggests adoption of plaintiff-friendly California approach rather than more restrictive alternatives. Sequential development shows clear progression toward enhanced consumer protection.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Texas",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 134,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Greenman v. Yuba Power Products (1963)", "Barker v. Lull Engineering (1978)", "Cronin v. J.B.E. Olson Corp. (1972)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Davis v. Social Media Inc., a First Amendment case. The court cited New York Times v. Sullivan (1964), Gertz v. Robert Welch (1974), and Hustler Magazine v. Falwell (1988). Explain how each precedent supports the analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Davis v. Social Media Inc. First Amendment Case: **New York Times v. Sullivan (1964)**: *How it helps*: Supreme Court case establishing 'actual malice' standard for public officials in defamation cases. Requires proof that defendant acted with knowledge of falsity or reckless disregard for truth. Provides heightened protection for speech about public officials. Creates constitutional floor protecting robust debate on public issues. *Why cited*: **Constitutional foundation** - Landmark decision constitutionalizing defamation law. Essential precedent for any case involving speech about public figures or matters of public concern. **Gertz v. Robert Welch (1974)**: *How it helps*: Supreme Court case extending First Amendment protection to speech about private individuals involved in public controversies. Establishes different standards based on plaintiff's public/private status. Requires at least negligence standard for private figure plaintiffs. Balances free speech against reputation protection. *Why cited*: **Standard determination** - Critical for determining what standard applies based on Davis's public/private status. Provides framework for analyzing different levels of constitutional protection. **Hustler Magazine v. Falwell (1988)**: *How it helps*: Supreme Court case extending actual malice standard to intentional infliction of emotional distress claims by public figures. Prevents circumvention of Sullivan standard through alternative tort theories. Provides broad protection for satirical and critical speech about public figures. *Why cited*: **Theory expansion** - Prevents plaintiff from avoiding First Amendment protection by using non-defamation tort theories. Ensures comprehensive free speech protection across multiple legal theories. **Precedent Interaction**: Sullivan establishes basic actual malice framework, Gertz creates public/private distinction with different standards, Hustler prevents evasion through alternative torts. Together they create comprehensive First Amendment defense: (1) Determine plaintiff's status (Gertz), (2) Apply appropriate standard (Sullivan), (3) Block alternative theories (Hustler). Sequential analysis maximizes speech protection.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "constitutional_law"
        },
        "extra_info": {
            "split": "train",
            "index": 135,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["New York Times v. Sullivan (1964)", "Gertz v. Robert Welch (1974)", "Hustler Magazine v. Falwell (1988)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Martinez v. City Bank, a contract interpretation case in New York. The court cited Frigaliment Importing Co. v. B.N.S. International Sales (1960), W.W.W. Associates v. Giancontieri (1990), and Raffles v. Wichelhaus (1864). How does each precedent help?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Martinez v. City Bank Contract Interpretation Case: **Frigaliment Importing Co. v. B.N.S. International Sales (1960)**: *How it helps*: Second Circuit case establishing framework for resolving contractual ambiguity. Creates hierarchy of interpretation methods: (1) parties' conduct and course of dealing, (2) industry custom and usage, (3) dictionary definitions. Provides systematic approach when contract terms have multiple meanings. *Why cited*: **Interpretation methodology** - Leading federal case on contract interpretation in New York. Provides structured approach for resolving ambiguous contract language in Martinez case. **W.W.W. Associates v. Giancontieri (1990)**: *How it helps*: New York Court of Appeals case establishing that parol evidence is inadmissible when contract language is unambiguous. Reinforces 'four corners' rule - court must determine meaning from contract text alone if clear. Limits ability to introduce extrinsic evidence of parties' intent. *Why cited*: **New York controlling precedent** - Binding authority on parol evidence rule in New York. Determines whether court can consider evidence beyond contract text to resolve interpretation disputes. **Raffles v. Wichelhaus (1864)**: *How it helps*: English case establishing mutual mistake doctrine when parties had different meanings for same contract term ('Peerless' ship). Creates framework for voiding contracts when fundamental misunderstanding exists. Demonstrates that contracts can fail for lack of meeting of minds. *Why cited*: **Classic precedent** - Historical foundation for mutual mistake doctrine. Provides theoretical basis for voiding contract if parties had fundamentally different understandings of key terms. **Precedent Hierarchy**: Raffles provides theoretical foundation for mutual mistake, Giancontieri establishes New York's restrictive approach to parol evidence, Frigaliment offers practical interpretation methodology. Together they create analytical framework: (1) Is contract language ambiguous? (Giancontieri), (2) If ambiguous, how to interpret? (Frigaliment), (3) If fundamental misunderstanding, void for mistake? (Raffles). Sequential analysis covers full range of interpretation issues.",
            "task_type": "precedent_analysis",
            "jurisdiction": "New_York",
            "legal_domain": "contract_law"
        },
        "extra_info": {
            "split": "train",
            "index": 136,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Frigaliment Importing Co. v. B.N.S. International Sales (1960)", "W.W.W. Associates v. Giancontieri (1990)", "Raffles v. Wichelhaus (1864)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Thompson v. Software Corp., a trade secret case. The court cited E.I. du Pont de Nemours & Co. v. Christopher (1970), Uniform Trade Secrets Act provisions, and Religious Technology Center v. Netcom (1995). How does each authority support the case?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Thompson v. Software Corp. Trade Secret Case: **E.I. du Pont de Nemours & Co. v. Christopher (1970)**: *How it helps*: Fifth Circuit case establishing that industrial espionage violates trade secret law even without traditional confidential relationship. Held that aerial photography of construction site to discover trade secrets constitutes improper means. Expands trade secret protection beyond breach of confidence to include any improper acquisition method. *Why cited*: **Improper means precedent** - Leading case defining what constitutes 'improper means' for trade secret acquisition. Critical if Thompson case involves unauthorized access to confidential information through technological means. **Uniform Trade Secrets Act (UTSA)**: *How it helps*: Model statute adopted by most states providing comprehensive trade secret protection framework. Defines trade secrets, improper means, and available remedies. Creates consistent national approach to trade secret law. Provides injunctive relief and damages remedies. *Why cited*: **Statutory framework** - Controlling state law (if adopted in jurisdiction) governing trade secret claims. Provides definitions, elements, and remedies for Thompson's claims. **Religious Technology Center v. Netcom (1995)**: *How it helps*: Northern District of California case applying trade secret law to internet and computer technology contexts. Addresses how traditional trade secret concepts apply to digital information and online dissemination. Considers unique aspects of protecting confidential information in electronic format. *Why cited*: **Technology application** - Precedent showing how trade secret law applies to software and digital contexts. Relevant if Thompson case involves software code, algorithms, or other digital trade secrets. **Authority Synergy**: Du Pont establishes broad 'improper means' concept applicable to technological espionage, UTSA provides statutory framework with definitions and remedies, Netcom applies these concepts specifically to digital/software contexts. Together they create comprehensive protection for digital trade secrets: (1) UTSA defines what qualifies as trade secret, (2) Du Pont establishes improper acquisition theory, (3) Netcom shows application to software industry. Framework covers both legal theory and practical application to Thompson's software industry context.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 137,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["E.I. du Pont de Nemours & Co. v. Christopher (1970)", "Religious Technology Center v. Netcom (1995)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Garcia v. Metro Hospital, a negligence case in Florida. The court cited Palsgraf v. Long Island Railroad (1928), Wagon Mound No. 1 (1961), and McCain v. Florida Power Corp. (1992). Explain how each precedent supports the negligence analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Garcia v. Metro Hospital Negligence Case: **Palsgraf v. Long Island Railroad (1928)**: *How it helps*: New York Court of Appeals case (Cardozo opinion) establishing foreseeability test for determining scope of duty in negligence. Holds that defendant owes duty only to foreseeable plaintiffs within zone of danger. Creates framework for limiting liability based on relationship between defendant's conduct and plaintiff's harm. *Why cited*: **Duty/proximate cause foundation** - Leading precedent on duty and foreseeability in negligence law. Essential for determining whether Metro Hospital owed duty to Garcia and whether Garcia was foreseeable plaintiff. **Wagon Mound No. 1 (1961)**: *How it helps*: Privy Council case establishing that damages must be reasonably foreseeable consequence of defendant's negligent act. Limits liability to foreseeable types of harm, not just foreseeable plaintiffs. Replaces direct causation test with foreseeability analysis for damages. *Why cited*: **Damages foreseeability** - Key precedent limiting recoverable damages to foreseeable consequences. Critical for determining which of Garcia's damages Metro Hospital should have anticipated. **McCain v. Florida Power Corp. (1992)**: *How it helps*: Florida Supreme Court case applying foreseeability analysis in specific Florida context. Establishes how Florida courts analyze duty and proximate cause in negligence cases. Provides state-specific application of general negligence principles. May address liability standards for institutional defendants like hospitals. *Why cited*: **Florida controlling precedent** - Binding authority on negligence analysis in Florida courts. Ensures analysis follows Florida-specific approach to duty, breach, causation, and damages. **Precedent Interaction**: Palsgraf establishes foundational duty/foreseeability framework, Wagon Mound extends foreseeability to damages analysis, McCain provides Florida-specific application. Together they create complete negligence framework: (1) Did defendant owe duty to this plaintiff? (Palsgraf), (2) Was this type of harm foreseeable? (Wagon Mound), (3) How does Florida apply these principles? (McCain). Sequential analysis ensures comprehensive negligence evaluation under Florida law while incorporating fundamental common law principles.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Florida",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 138,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Palsgraf v. Long Island Railroad (1928)", "Wagon Mound No. 1 (1961)", "McCain v. Florida Power Corp. (1992)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Robinson v. Energy Corp., a shareholder derivative suit in Delaware. The court cited Aronson v. Lewis (1984), Zapata Corp. v. Maldonado (1981), and Rales v. Blasband (1993). How does each precedent support the procedural analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Robinson v. Energy Corp. Shareholder Derivative Suit: **Aronson v. Lewis (1984)**: *How it helps*: Delaware Supreme Court case establishing demand futility test for derivative suits. Creates two-prong analysis: (1) whether directors could have exercised independent business judgment, and (2) whether challenged transaction was product of valid business judgment. Provides framework for determining when shareholders must make demand on board before filing suit. *Why cited*: **Demand futility standard** - Controlling Delaware precedent for derivative suit procedure. Essential for determining whether Robinson must make demand on Energy Corp. board or whether demand would be futile. **Zapata Corp. v. Maldonado (1981)**: *How it helps*: Delaware Supreme Court case establishing two-step analysis for evaluating special litigation committee recommendations to dismiss derivative suits. Creates framework for judicial review of board decisions to terminate derivative litigation. Provides independent judicial review even when committee acts independently. *Why cited*: **Committee review framework** - Controls how Delaware courts evaluate board attempts to dismiss derivative suits. Critical if Energy Corp. board attempts to terminate Robinson's suit through special committee. **Rales v. Blasband (1993)**: *How it helps*: Delaware Supreme Court case clarifying demand futility analysis when derivative suit challenges board inaction rather than specific transaction. Establishes that Aronson test doesn't apply to failure-to-act cases. Creates alternative analysis focusing on whether board faces substantial likelihood of liability. *Why cited*: **Inaction-specific analysis** - Provides framework when derivative suit challenges board's failure to act rather than affirmative wrongdoing. May be relevant if Robinson challenges board's failure to pursue claims or oversight. **Procedural Strategy**: Aronson provides general demand futility test, Rales offers alternative for inaction cases, Zapata addresses committee dismissal attempts. Together they create comprehensive derivative suit roadmap: (1) Is demand required or futile? (Aronson/Rales), (2) If suit proceeds, can board dismiss through committee? (Zapata). Framework covers full derivative suit lifecycle from filing through potential dismissal, maximizing plaintiff's procedural advantages while respecting board authority.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Delaware",
            "legal_domain": "corporate_law"
        },
        "extra_info": {
            "split": "train",
            "index": 139,
            "difficulty": "advanced",
            "court_level": "state_trial",
            "case_citations": ["Aronson v. Lewis (1984)", "Zapata Corp. v. Maldonado (1981)", "Rales v. Blasband (1993)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Anderson v. Tech Startup, a non-compete case in California. The court cited Edwards v. Arthur Andersen (2008), Application Group v. Hunter Group (1998), and Whitewater West Industries v. Alleshouse (2004). How does each precedent help the analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Anderson v. Tech Startup Non-Compete Case: **Edwards v. Arthur Andersen (2008)**: *How it helps*: California Supreme Court case invalidating non-compete agreements under Business and Professions Code Section 16600. Establishes that California public policy strongly favors employee mobility and competition. Holds that non-compete clauses are void and unenforceable regardless of their reasonableness or geographic scope. Creates bright-line rule against restraints on trade. *Why cited*: **Controlling precedent** - Binding California authority establishing broad prohibition on non-compete agreements. Provides foundation for invalidating any non-compete restrictions against Anderson. **Application Group v. Hunter Group (1998)**: *How it helps*: California Court of Appeal case applying Edwards principle to narrow exceptions. Clarifies that even 'reasonable' non-compete agreements violate Section 16600. Addresses employer arguments that certain restraints should be permissible. May discuss trade secret protection as alternative to non-compete. *Why cited*: **Precedent application** - Shows how California courts apply Edwards in practice. Demonstrates rejection of employer arguments for exceptions to non-compete prohibition. **Whitewater West Industries v. Alleshouse (2004)**: *How it helps*: California Court of Appeal case addressing relationship between non-compete prohibition and trade secret protection. Clarifies that while non-competes are invalid, employers can still protect legitimate trade secrets through confidentiality agreements. Provides guidance on permissible vs. impermissible restraints. *Why cited*: **Boundary definition** - Helps distinguish invalid non-compete restraints from valid trade secret protection. Important for determining which restrictive provisions Tech Startup can enforce against Anderson. **California's Unique Approach**: Edwards establishes California's categorical rejection of non-competes, Application Group shows consistent judicial application, Whitewater defines boundaries with trade secret law. Together they create employee-favorable framework: (1) Non-competes are categorically void (Edwards), (2) No exceptions for 'reasonable' restraints (Application Group), (3) Trade secret protection remains available through other means (Whitewater). Analysis maximizes Anderson's mobility rights while preserving Tech Startup's legitimate confidentiality interests.",
            "task_type": "precedent_analysis",
            "jurisdiction": "California",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 140,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Edwards v. Arthur Andersen (2008)", "Application Group v. Hunter Group (1998)", "Whitewater West Industries v. Alleshouse (2004)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Wilson v. Insurance Co., a bad faith insurance case in Texas. The court cited Aranda v. Insurance Co. of North America (1994), Universe Life Insurance Co. v. Giles (1991), and Lynch v. Merrell-National Laboratories (1987). Explain how each precedent supports the bad faith analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Wilson v. Insurance Co. Bad Faith Insurance Case: **Aranda v. Insurance Co. of North America (1994)**: *How it helps*: Texas Supreme Court case establishing elements for insurance bad faith claims under Texas law. Requires showing that: (1) insurer had no reasonable basis for denial, and (2) insurer knew or should have known denial was unreasonable. Creates objective standard focusing on reasonableness of insurer's investigation and decision-making process. *Why cited*: **Texas controlling precedent** - Binding authority establishing bad faith elements in Texas. Provides framework for analyzing whether Insurance Co.'s conduct toward Wilson meets bad faith standards. **Universe Life Insurance Co. v. Giles (1991)**: *How it helps*: Texas Supreme Court case clarifying that bad faith requires more than mere negligence - insurer must act without reasonable basis and know the denial was unreasonable. Establishes heightened standard requiring conscious indifference or deliberate disregard. Protects insurers from bad faith liability for good faith mistakes. *Why cited*: **Standard clarification** - Defines the mental state required for bad faith liability. Critical for determining whether Insurance Co.'s conduct rises to level of bad faith or constitutes mere negligence. **Lynch v. Merrell-National Laboratories (1987)**: *How it helps*: Texas Supreme Court case establishing framework for punitive damages in cases involving conscious indifference or malice. Provides standards for when conduct warrants exemplary damages beyond compensatory relief. May address burden of proof and evidence requirements for punitive awards. *Why cited*: **Damages framework** - Relevant if Wilson seeks punitive damages for Insurance Co.'s bad faith conduct. Establishes standards for enhanced damages based on egregious conduct. **Texas Bad Faith Framework**: Aranda establishes basic bad faith elements, Universe clarifies heightened mental state requirement, Lynch provides punitive damages framework. Together they create Texas-specific approach: (1) Was denial unreasonable and did insurer know it? (Aranda), (2) Did conduct exceed mere negligence? (Universe), (3) If bad faith proven, are punitive damages warranted? (Lynch). Sequential analysis covers liability determination through damages calculation, ensuring comprehensive bad faith evaluation under Texas law.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Texas",
            "legal_domain": "insurance_law"
        },
        "extra_info": {
            "split": "train",
            "index": 141,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Aranda v. Insurance Co. of North America (1994)", "Universe Life Insurance Co. v. Giles (1991)", "Lynch v. Merrell-National Laboratories (1987)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Miller v. City Council, a municipal liability case under Section 1983. The court cited Monell v. Department of Social Services (1978), Canton v. Harris (1989), and Pembaur v. Cincinnati (1986). How does each precedent support the civil rights analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Miller v. City Council Section 1983 Municipal Liability Case: **Monell v. Department of Social Services (1978)**: *How it helps*: Supreme Court case establishing that municipalities can be sued under Section 1983 for constitutional violations. Overrules Monroe v. Pape's municipal immunity doctrine. Requires showing that constitutional violation resulted from official municipal policy or custom. Distinguishes between individual officer liability and municipal entity liability. *Why cited*: **Municipal liability foundation** - Essential precedent establishing that cities can be Section 1983 defendants. Provides framework for holding City Council liable for constitutional violations through official policies. **Canton v. Harris (1989)**: *How it helps*: Supreme Court case establishing 'deliberate indifference' standard for municipal liability based on failure to train employees. Requires showing that: (1) training was inadequate, (2) city was deliberately indifferent to constitutional violations, (3) inadequate training caused the violation. Creates pathway for municipal liability through training deficiencies. *Why cited*: **Training liability theory** - Provides framework for municipal liability based on inadequate training or supervision. Relevant if Miller's constitutional violation resulted from City Council's failure to properly train officers or employees. **Pembaur v. Cincinnati (1986)**: *How it helps*: Supreme Court case clarifying that single decision by municipal policymaker can establish official policy for Section 1983 liability. Eliminates need to show pattern of violations - one decision by person with final authority suffices. Defines who qualifies as municipal policymaker with authority to create official policy. *Why cited*: **Single-incident liability** - Allows municipal liability based on one policy decision rather than requiring pattern of violations. Critical if Miller's case involves single decision by City Council member with policymaking authority. **Municipal Liability Framework**: Monell establishes basic municipal liability under Section 1983, Canton provides deliberate indifference pathway through training failures, Pembaur allows liability from single policymaker decision. Together they create comprehensive municipal liability theories: (1) Official policy violation (Monell), (2) Deliberate indifference to training needs (Canton), (3) Single decision by policymaker (Pembaur). Framework maximizes chances of municipal liability while respecting constitutional requirements for entity responsibility.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "civil_rights_law"
        },
        "extra_info": {
            "split": "train",
            "index": 142,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Monell v. Department of Social Services (1978)", "Canton v. Harris (1989)", "Pembaur v. Cincinnati (1986)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Taylor v. University Hospital, a wrongful termination case in Illinois. The court cited Palmateer v. International Harvester Co. (1981), Kelsay v. Motorola (1995), and Zimmerman v. Buchheit of Sparta (2011). How does each precedent support the retaliatory discharge analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Taylor v. University Hospital Wrongful Termination Case: **Palmateer v. International Harvester Co. (1981)**: *How it helps*: Illinois Supreme Court case establishing public policy exception to at-will employment. Creates cause of action for retaliatory discharge when termination violates clear mandate of public policy. Establishes four situations warranting protection: (1) refusing to violate law, (2) performing public duty, (3) exercising statutory right, (4) reporting violations. *Why cited*: **Foundation precedent** - Controlling Illinois authority creating wrongful termination cause of action. Essential for establishing that Taylor can sue despite at-will employment status. **Kelsay v. Motorola (1995)**: *How it helps*: Illinois Supreme Court case clarifying public policy exception requirements. Establishes that public policy must be clearly mandated and well-defined, not merely preferred or encouraged. Creates heightened standard requiring specific legal mandate rather than general policy preference. Limits scope of public policy exception. *Why cited*: **Standard refinement** - Clarifies what constitutes sufficient public policy to support wrongful discharge claim. Critical for determining whether Taylor's conduct falls within protected public policy categories. **Zimmerman v. Buchheit of Sparta (2011)**: *How it helps*: Illinois Supreme Court case applying public policy analysis to specific employment context. May address healthcare-related public policies or professional obligations. Provides modern application of Palmateer/Kelsay framework to contemporary workplace issues. Shows how courts analyze public policy sources and clarity requirements. *Why cited*: **Contemporary application** - Demonstrates current Illinois approach to public policy analysis. Relevant for applying established framework to Taylor's specific situation at University Hospital. **Illinois Wrongful Discharge Framework**: Palmateer creates basic public policy exception, Kelsay establishes heightened clarity requirements, Zimmerman provides modern application guidance. Together they create analytical framework: (1) Was Taylor's conduct protected by public policy? (Palmateer), (2) Is the public policy clear and well-defined? (Kelsay), (3) How do courts apply these standards in practice? (Zimmerman). Sequential analysis ensures comprehensive evaluation of Taylor's wrongful discharge claim under current Illinois law while respecting limitations on at-will employment exceptions.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Illinois",
            "legal_domain": "employment_law"
        },
        "extra_info": {
            "split": "train",
            "index": 143,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Palmateer v. International Harvester Co. (1981)", "Kelsay v. Motorola (1995)", "Zimmerman v. Buchheit of Sparta (2011)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Parker v. Construction Co., a premises liability slip-and-fall case in Georgia. The court cited Alterman Foods v. Ligon (1985), Lau's Corp. v. Haskins (1995), and Robinson v. Kroger Co. (1990). Explain how each precedent supports the premises liability analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Parker v. Construction Co. Premises Liability Case: **Alterman Foods v. Ligon (1985)**: *How it helps*: Georgia Supreme Court case establishing that premises liability requires showing: (1) defendant had superior knowledge of hazard, or (2) hazard was present for sufficient time that defendant should have discovered it through reasonable inspection. Creates framework for analyzing notice requirements in slip-and-fall cases. Balances property owner duties with reasonable expectations. *Why cited*: **Georgia controlling precedent** - Binding authority establishing premises liability elements in Georgia. Provides framework for analyzing whether Construction Co. had actual or constructive notice of dangerous condition. **Lau's Corp. v. Haskins (1995)**: *How it helps*: Georgia Supreme Court case clarifying constructive notice analysis for premises liability. Establishes that plaintiff must prove hazard existed for sufficient time to give reasonable property owner opportunity to discover and remedy it. Creates temporal requirement for constructive notice - brief existence of hazard insufficient for liability. *Why cited*: **Notice timing standard** - Defines how long hazard must exist before constructive notice applies. Critical for determining whether dangerous condition was present long enough to charge Construction Co. with notice. **Robinson v. Kroger Co. (1990)**: *How it helps*: Georgia Court of Appeals case applying premises liability analysis to specific factual contexts. May address burden of proof, evidence requirements, or application of notice standards to particular types of hazards. Provides practical application of Alterman/Lau framework to similar slip-and-fall scenarios. *Why cited*: **Factual application** - Shows how Georgia courts apply premises liability standards in practice. Provides precedent for similar factual circumstances to Parker's case. **Georgia Premises Liability Structure**: Alterman establishes basic notice requirements (actual or constructive), Lau clarifies timing requirements for constructive notice, Robinson provides factual application guidance. Together they create comprehensive premises liability analysis: (1) Did defendant have actual knowledge of hazard? (Alterman), (2) If constructive notice claimed, was hazard present long enough? (Lau), (3) How do courts apply these standards to similar facts? (Robinson). Framework balances property owner responsibilities with reasonable limits on liability, ensuring Parker meets Georgia's specific requirements for premises liability claims.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Georgia",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 144,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Alterman Foods v. Ligon (1985)", "Lau's Corp. v. Haskins (1995)", "Robinson v. Kroger Co. (1990)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Stewart v. Pharmaceutical Inc., a drug product liability case. The court cited Daubert v. Merrell Dow Pharmaceuticals (1993), Kumho Tire v. Carmichael (1999), and Joiner v. General Electric (1997). How does each precedent support the expert testimony analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Stewart v. Pharmaceutical Inc. Expert Testimony in Product Liability: **Daubert v. Merrell Dow Pharmaceuticals (1993)**: *How it helps*: Supreme Court case establishing framework for admitting scientific expert testimony under Federal Rule of Evidence 702. Replaces Frye 'general acceptance' test with reliability-based analysis. Creates five-factor test: (1) testability, (2) peer review, (3) error rates, (4) standards, (5) general acceptance. Makes trial judges 'gatekeepers' for scientific evidence. *Why cited*: **Expert admissibility foundation** - Controlling precedent for evaluating scientific expert testimony in federal court. Essential for determining whether Stewart's medical/scientific experts can testify about drug causation. **Joiner v. General Electric (1997)**: *How it helps*: Supreme Court case establishing that trial court's Daubert rulings are reviewed for abuse of discretion, not de novo. Clarifies that courts can exclude expert testimony when there's too great analytical gap between data and conclusions. Emphasizes that trial judges have broad discretion in reliability determinations. *Why cited*: **Judicial discretion standard** - Establishes deferential appellate review of expert testimony rulings. Critical for understanding trial court's authority to exclude unreliable expert opinions in Stewart case. **Kumho Tire v. Carmichael (1999)**: *How it helps*: Supreme Court case extending Daubert reliability analysis to all expert testimony, not just scientific evidence. Establishes that trial courts must ensure all expert testimony is reliable, including technical and experience-based expertise. Provides flexibility in applying reliability factors based on type of expertise. *Why cited*: **Non-scientific expert coverage** - Extends Daubert to all types of expert testimony. Relevant if Stewart case involves non-scientific experts like pharmacists, physicians with clinical experience, or industry experts. **Expert Testimony Trilogy**: Daubert establishes reliability framework for scientific evidence, Joiner provides discretion standard for trial courts, Kumho extends analysis to all expert testimony. Together they create comprehensive expert evidence analysis: (1) Is testimony reliable under appropriate factors? (Daubert/Kumho), (2) Does trial court have discretion in reliability determination? (Joiner), (3) Do reliability requirements apply to all expert types? (Kumho). Framework ensures rigorous evaluation of expert testimony while respecting judicial discretion and covering all expertise categories relevant to pharmaceutical product liability.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "evidence_law"
        },
        "extra_info": {
            "split": "train",
            "index": 145,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Daubert v. Merrell Dow Pharmaceuticals (1993)", "Kumho Tire v. Carmichael (1999)", "Joiner v. General Electric (1997)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Adams v. Real Estate Corp., a landlord-tenant case in Ohio. The court cited Shroades v. Rental Property Management (1987), Phale v. Keim (1981), and Blackwell v. Del Bosco (1979). How does each precedent help the habitability analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Adams v. Real Estate Corp. Landlord-Tenant Habitability Case: **Shroades v. Rental Property Management (1987)**: *How it helps*: Ohio Supreme Court case establishing implied warranty of habitability in residential leases. Requires landlords to maintain rental property in habitable condition regardless of lease terms. Creates tenant remedies including repair and deduct, withholding rent, and termination for material breaches. Represents Ohio's adoption of modern landlord-tenant law. *Why cited*: **Ohio controlling precedent** - Binding authority establishing warranty of habitability in Ohio. Provides foundation for Adams' claims against Real Estate Corp. for uninhabitable conditions. **Phale v. Keim (1981)**: *How it helps*: Ohio Supreme Court case defining what constitutes 'uninhabitable' conditions under warranty of habitability. Establishes standards for determining when rental property fails to meet habitability requirements. May address specific types of defects or conditions that violate habitability warranty. Provides guidance on severity thresholds. *Why cited*: **Habitability standards** - Defines what conditions violate warranty of habitability. Critical for determining whether Adams' rental unit conditions meet legal standards for uninhabitability. **Blackwell v. Del Bosco (1979)**: *How it helps*: Ohio case addressing tenant remedies for habitability violations. May establish framework for damages, rent withholding procedures, or termination rights when landlord breaches warranty of habitability. Provides practical guidance on tenant options when habitability violations occur. *Why cited*: **Remedies framework** - Establishes available remedies for habitability violations. Important for determining what relief Adams can obtain from Real Estate Corp. for breach of warranty. **Ohio Habitability Framework**: Shroades establishes implied warranty of habitability, Phale defines uninhabitable conditions, Blackwell provides remedies for violations. Together they create comprehensive tenant protection: (1) Does warranty of habitability apply? (Shroades), (2) Do conditions violate habitability standards? (Phale), (3) What remedies are available for violations? (Blackwell). Sequential analysis covers full habitability claim from establishing warranty through obtaining relief, maximizing tenant protection under Ohio law while providing clear standards for landlord obligations.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Ohio",
            "legal_domain": "property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 146,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Shroades v. Rental Property Management (1987)", "Phale v. Keim (1981)", "Blackwell v. Del Bosco (1979)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Cooper v. Media Corporation, a privacy tort case. The court cited Prosser's four privacy torts framework, Cox Broadcasting v. Cohn (1975), and Time Inc. v. Hill (1967). Explain how each authority supports the privacy analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Cooper v. Media Corporation Privacy Tort Case: **Prosser's Four Privacy Torts Framework**: *How it helps*: Dean William Prosser's influential law review article identifying four distinct privacy tort categories: (1) intrusion upon seclusion, (2) public disclosure of private facts, (3) false light invasion of privacy, (4) appropriation of likeness. Provides analytical structure adopted by most courts for privacy tort analysis. Creates framework for categorizing different types of privacy violations. *Why cited*: **Conceptual framework** - Foundational scholarly work organizing privacy tort law. Essential for determining which privacy tort theory applies to Cooper's claims against Media Corporation. **Cox Broadcasting v. Cohn (1975)**: *How it helps*: Supreme Court case establishing First Amendment limitations on privacy tort claims. Holds that truthful publication of information obtained from public records cannot be basis for privacy tort liability. Creates constitutional protection for news media reporting matters of public record. Balances privacy rights against free speech protections. *Why cited*: **Constitutional limitations** - Establishes First Amendment constraints on privacy tort liability. Critical if Media Corporation's conduct involved publishing truthful information from public sources. **Time Inc. v. Hill (1967)**: *How it helps*: Supreme Court case applying First Amendment actual malice standard to false light privacy claims involving matters of public interest. Requires public figure plaintiffs to prove actual malice (knowledge of falsity or reckless disregard). Provides heightened protection for media when reporting on newsworthy events. *Why cited*: **False light standards** - Establishes constitutional requirements for false light privacy claims. Relevant if Cooper's case involves false light theory and matters of public concern. **Privacy Law Structure**: Prosser framework categorizes types of privacy violations, Cox Broadcasting establishes constitutional limits on public information, Time provides actual malice standard for false light claims. Together they create comprehensive privacy analysis: (1) Which privacy tort applies? (Prosser), (2) Are there constitutional barriers to liability? (Cox Broadcasting), (3) If false light claimed, what standard applies? (Time). Framework balances privacy protection with First Amendment rights, ensuring proper categorization and constitutional compliance in Cooper's case against Media Corporation.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 147,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["Cox Broadcasting v. Cohn (1975)", "Time Inc. v. Hill (1967)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Morris v. Banking Corp., a consumer protection case under state UDAP statutes. The court cited FTC v. Colgate-Palmolive (1965), FTC Policy Statement on Deception (1983), and Lesnick v. Hollingsworth & Vose (1994). How does each authority support the deceptive practices analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Morris v. Banking Corp. Consumer Protection UDAP Case: **FTC v. Colgate-Palmolive (1965)**: *How it helps*: Supreme Court case establishing FTC authority to regulate deceptive advertising practices. Holds that FTC can prohibit practices that have capacity to deceive, even without proof of actual deception. Creates framework for analyzing whether business practices are inherently deceptive. Provides foundation for federal consumer protection enforcement. *Why cited*: **Deception precedent** - Establishes federal standards for determining deceptive practices. Relevant because many state UDAP statutes follow FTC Act interpretation, making federal precedent persuasive for state law analysis. **FTC Policy Statement on Deception (1983)**: *How it helps*: Official FTC guidance defining when practices constitute deception under federal law. Establishes three-part test: (1) representation likely to mislead consumers, (2) from perspective of reasonable consumer, (3) material to consumer decision-making. Provides authoritative interpretation of deception standards used by courts and agencies. *Why cited*: **Authoritative definition** - FTC's formal interpretation of deception standards. Important because state UDAP statutes often incorporate or parallel FTC Act standards, making FTC interpretation relevant to state law claims. **Lesnick v. Hollingsworth & Vose (1994)**: *How it helps*: Court case applying deception analysis to specific business practices in private litigation context. Shows how deception standards apply in consumer lawsuits rather than just agency enforcement. May address damages, causation, or other elements specific to private UDAP claims. Provides practical application of deception principles. *Why cited*: **Private litigation application** - Demonstrates how deception standards apply in consumer lawsuits. Relevant for Morris's private UDAP claim against Banking Corp., showing how federal deception concepts translate to state consumer protection cases. **Consumer Protection Framework**: Colgate-Palmolive establishes federal deception precedent, FTC Policy Statement provides authoritative definition, Lesnick shows private litigation application. Together they create deception analysis structure: (1) What constitutes deceptive practice? (Colgate-Palmolive), (2) How is deception defined and analyzed? (FTC Policy Statement), (3) How do these standards apply in private lawsuits? (Lesnick). Framework ensures comprehensive deception analysis while bridging federal precedent with state UDAP enforcement in Morris's consumer protection claim.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "consumer_law"
        },
        "extra_info": {
            "split": "train",
            "index": 148,
            "difficulty": "intermediate",
            "court_level": "federal_district",
            "case_citations": ["FTC v. Colgate-Palmolive (1965)", "Lesnick v. Hollingsworth & Vose (1994)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm analyzing Hughes v. Transportation Co., a personal injury case involving res ipsa loquitur. The court cited Byrne v. Boadle (1863), Ybarra v. Spangard (1944), and McDougald v. Perry (1998). How does each precedent support the res ipsa analysis?"
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Hughes v. Transportation Co. Res Ipsa Loquitur Case: **Byrne v. Boadle (1863)**: *How it helps*: English case establishing foundational res ipsa loquitur doctrine (barrel falling from defendant's window). Creates principle that some accidents speak for themselves and permit inference of negligence without direct proof. Establishes that circumstantial evidence can be sufficient when accident type ordinarily results from negligence. Historical foundation for res ipsa doctrine. *Why cited*: **Doctrine foundation** - Classic case establishing res ipsa loquitur principle. Provides theoretical basis for allowing negligence inference from circumstantial evidence when Hughes lacks direct proof of Transportation Co.'s negligence. **Ybarra v. Spangard (1944)**: *How it helps*: California Supreme Court case expanding res ipsa loquitur to cases involving multiple defendants where plaintiff cannot identify specific responsible party. Allows res ipsa application when plaintiff was unconscious or otherwise unable to observe negligent conduct. Creates framework for shifting burden to defendants to explain their conduct. *Why cited*: **Multiple defendant application** - Shows how res ipsa applies when specific negligent actor cannot be identified. Relevant if Hughes case involves multiple defendants or if Hughes was unable to observe Transportation Co.'s specific negligent conduct. **McDougald v. Perry (1998)**: *How it helps*: Florida Supreme Court case establishing modern three-part test for res ipsa loquitur: (1) accident must be of type that ordinarily doesn't occur without negligence, (2) instrumentality causing injury must be under defendant's exclusive control, (3) plaintiff must not have contributed to accident. Provides contemporary framework for res ipsa analysis. *Why cited*: **Modern framework** - Establishes current analytical structure for res ipsa claims. Critical for determining whether Hughes can satisfy all elements necessary for res ipsa application against Transportation Co. **Res Ipsa Evolution**: Byrne establishes foundational doctrine allowing negligence inference, Ybarra expands application to complex defendant scenarios, McDougald provides modern three-element framework. Together they create comprehensive res ipsa analysis: (1) Can negligence be inferred from circumstances? (Byrne), (2) Does doctrine apply with multiple parties or limited plaintiff knowledge? (Ybarra), (3) Are all modern elements satisfied? (McDougald). Framework maximizes Hughes's ability to prove negligence through circumstantial evidence while ensuring proper application of res ipsa requirements.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Florida",
            "legal_domain": "tort_law"
        },
        "extra_info": {
            "split": "train",
            "index": 149,
            "difficulty": "intermediate",
            "court_level": "state_trial",
            "case_citations": ["Byrne v. Boadle (1863)", "Ybarra v. Spangard (1944)", "McDougald v. Perry (1998)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
            {
                "role": "user",
                "content": "I'm reviewing Clark v. Biotech Inc., a patent infringement case. The court cited Markman v. Westview Instruments (1996), All Elements Rule from Pennwalt Corp. v. Durand-Wayland (1987), and Doctrine of Equivalents from Graver Tank v. Linde Air Products (1950). Explain how each precedent supports the infringement analysis."
            }
        ],
        "reward_model": {
            "style": "multi_task_legal",
            "ground_truth": "Precedent Analysis for Clark v. Biotech Inc. Patent Infringement Case: **Markman v. Westview Instruments (1996)**: *How it helps*: Supreme Court case establishing that claim construction is question of law for the court, not jury. Requires judge to interpret patent claim language and scope before infringement analysis. Creates framework for determining what the patent actually covers before comparing to accused device. Ensures uniform claim interpretation across different cases. *Why cited*: **Claim construction authority** - Controlling precedent requiring court to interpret Clark's patent claims before analyzing infringement. Essential first step in any patent infringement case to define scope of patent protection. **Pennwalt Corp. v. Durand-Wayland (1987) - All Elements Rule**: *How it helps*: Federal Circuit case establishing that patent infringement requires accused device to contain every element of at least one patent claim, either literally or equivalently. Creates requirement that plaintiff prove each claim element is present in defendant's device. Prevents infringement finding when any claim element is missing. *Why cited*: **Infringement requirement** - Establishes that Clark must prove Biotech's device contains every element of asserted patent claims. Critical framework for systematic infringement analysis ensuring comprehensive element-by-element comparison. **Graver Tank v. Linde Air Products (1950) - Doctrine of Equivalents**: *How it helps*: Supreme Court case establishing doctrine of equivalents for patent infringement when accused device performs substantially same function, in substantially same way, to achieve substantially same result. Allows infringement finding even without literal claim element correspondence. Provides alternative infringement theory when literal infringement cannot be proven. *Why cited*: **Equivalents theory** - Provides pathway for infringement when Biotech's device doesn't literally meet every claim element but performs substantially equivalent functions. Expands infringement analysis beyond strict literal interpretation. **Patent Infringement Framework**: Markman requires claim construction first, All Elements Rule establishes comprehensive infringement requirements, Doctrine of Equivalents provides alternative infringement theory. Together they create systematic infringement analysis: (1) What do patent claims mean? (Markman), (2) Does accused device contain every claim element literally? (All Elements Rule), (3) If not literal infringement, are differences insubstantial? (Doctrine of Equivalents). Framework ensures thorough infringement evaluation while balancing patent protection with accused infringer's rights.",
            "task_type": "precedent_analysis",
            "jurisdiction": "Federal",
            "legal_domain": "intellectual_property_law"
        },
        "extra_info": {
            "split": "train",
            "index": 150,
            "difficulty": "advanced",
            "court_level": "federal_district",
            "case_citations": ["Markman v. Westview Instruments (1996)", "Pennwalt Corp. v. Durand-Wayland (1987)", "Graver Tank v. Linde Air Products (1950)"],
            "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC judicial reasoning to analyze this negligence case: A grocery store failed to clean up a grape that had been on the floor for 2 hours. Mrs. Johnson, a 65-year-old customer, slipped on the grape while reaching for milk, suffering a broken hip. Security footage shows 15 customers walked past the grape before Mrs. Johnson's fall. The grape was in a high-traffic area near the dairy section. Analyze whether the store is liable for negligence."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Grape on floor for 2 hours, (2) Mrs. Johnson slipped and broke hip, (3) 15 customers passed grape before incident, (4) High-traffic dairy section location, (5) Store failed to clean despite opportunity. ISSUE: Whether store breached duty of reasonable care. RULE: Property owners owe invitees duty of reasonable care including regular inspection and prompt cleanup. APPLICATION: 2-hour duration + high traffic + 15 passersby establish constructive notice. CONCLUSION: Store liable. COUNTERFACTUAL: If grape present only 10 minutes, likely no liability due to lack of notice.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "tort_law"
        },
        "extra_info": {
        "split": "train",
        "index": 151,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Ortega v. Kmart Corp., 26 Cal. 4th 1200 (2001)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this contract dispute: TechCorp agreed to deliver custom software to MedClinic by March 1st for $500,000. The contract included a liquidated damages clause of $10,000 per day for late delivery. TechCorp delivered functional software on March 15th, but MedClinic claims it was missing key features and refuses to pay. TechCorp argues the software meets specifications and demands payment minus applicable late fees. What is the likely judicial reasoning and outcome?"
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Software delivery due March 1st, (2) $500,000 contract price, (3) $10,000/day liquidated damages, (4) Actual delivery March 15th, (5) MedClinic claims missing features, (6) TechCorp claims compliance. ISSUE: Whether substantial performance occurred and damages calculation. RULE: Substantial performance excuses minor deviations; material breach justifies non-payment. APPLICATION: Must examine specifications vs. delivered features. If substantial performance: payment due minus $140,000 late fees. If material breach: no payment due. CONCLUSION: Depends on feature analysis. ALTERNATIVE: If features minor, TechCorp wins $360,000; if material, MedClinic owes nothing.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "contract_law"
        },
        "extra_info": {
        "split": "train",
        "index": 152,
        "difficulty": "advanced",
        "court_level": "state_appellate",
        "case_citations": ["Jacob & Youngs v. Kent, 230 N.Y. 239 (1921)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this Fourth Amendment case: Police received an anonymous tip that John was selling drugs from his apartment. Without a warrant, they set up surveillance and observed heavy foot traffic. After seeing a known drug user exit the building, they knocked on John's door. When John answered, they smelled marijuana and pushed past him, discovering cocaine on a table. John moves to suppress the evidence. Analyze the constitutional issues using judicial reasoning."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Anonymous tip about drug sales, (2) Police surveillance showing foot traffic, (3) Known user exited building, (4) Police smelled marijuana at door, (5) Officers pushed past defendant, (6) Cocaine in plain view. ISSUE: Whether warrantless entry and search violate Fourth Amendment. RULE: Warrantless home entries require consent, exigent circumstances, or hot pursuit. Plain smell alone insufficient for entry. APPLICATION: No consent shown; pushing past = unlawful entry; plain view doctrine inapplicable to illegally obtained evidence. CONCLUSION: Evidence suppressed due to unlawful entry. COUNTERFACTUAL: If John consented to entry, evidence admissible under plain view.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "constitutional_law"
        },
        "extra_info": {
        "split": "train",
        "index": 153,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Payton v. New York, 445 U.S. 573 (1980)", "Kentucky v. King, 563 U.S. 452 (2011)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC: Sarah and Tom signed a real estate purchase agreement for a $800,000 house with a 30-day closing period. The contract required Tom to secure financing within 15 days or forfeit his $50,000 earnest money. Tom applied for a loan on day 5, was pre-approved on day 10, but final underwriting took until day 18 due to the lender's error. Tom notified Sarah immediately of the delay. Sarah claims the contract is breached and keeps the earnest money. Tom argues he substantially complied and the delay was beyond his control."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) $800,000 purchase price, (2) 30-day closing, 15-day financing deadline, (3) $50,000 earnest money at risk, (4) Tom applied day 5, pre-approved day 10, (5) Final approval day 18 due to lender error, (6) Tom notified Sarah of delay. ISSUE: Whether time-of-essence clause is strictly enforced or excused by impossibility. RULE: Time-of-essence clauses are strictly construed, but impossibility/frustration may excuse delay. APPLICATION: Lender delay beyond Tom's control; good faith efforts shown; minimal prejudice to Sarah. CONCLUSION: Tom likely keeps earnest money due to impossibility doctrine. ALTERNATIVE ARGUMENT: If court applies strict compliance, Sarah wins forfeiture.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "property_law"
        },
        "extra_info": {
        "split": "train",
        "index": 154,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Restatement (Second) of Contracts § 261"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this employment discrimination case: Maria, a Hispanic woman, worked as a financial analyst for 5 years with excellent performance reviews. When a senior analyst position opened, she applied along with 3 other candidates: two white males and one Asian female. The hiring manager, who had previously made comments about 'cultural fit,' selected David, a white male with 2 years less experience and lower performance ratings. Maria filed a Title VII discrimination claim. Analyze the judicial reasoning for her case."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Maria: Hispanic, 5 years experience, excellent reviews, (2) Applied for senior analyst promotion, (3) Competing with 2 white males, 1 Asian female, (4) David selected: white male, 3 years experience, lower ratings, (5) Hiring manager made 'cultural fit' comments. ISSUE: Whether Maria can establish prima facie discrimination case. RULE: McDonnell Douglas burden-shifting: plaintiff must show protected class, qualification, adverse action, circumstantial evidence of discrimination. APPLICATION: Hispanic = protected class; 5-year experience + excellent reviews = qualification; promotion denial = adverse action; less qualified white male selected + 'cultural fit' comments = circumstantial evidence. CONCLUSION: Prima facie case established; burden shifts to employer. CRITICAL FACT: 'Cultural fit' comment suggests discriminatory animus.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "employment_law"
        },
        "extra_info": {
        "split": "train",
        "index": 155,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["McDonnell Douglas Corp. v. Green, 411 U.S. 792 (1973)", "Texas Dep't of Cmty. Affairs v. Burdine, 450 U.S. 248 (1981)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this products liability case: XYZ Corp manufactured a power saw with a safety guard. Worker Joe removed the guard to make faster cuts, despite clear warnings in the manual. The saw blade shattered during use, sending fragments into Joe's eye, causing permanent vision loss. Joe sues XYZ Corp for defective design, claiming the saw should have been designed to prevent operation without the guard. XYZ argues Joe's modification caused his injury."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) XYZ manufactured power saw with safety guard, (2) Joe removed guard for faster cutting, (3) Manual contained clear warnings, (4) Blade shattered during use, (5) Fragments caused permanent eye injury, (6) Joe sues for defective design. ISSUE: Whether product was unreasonably dangerous despite user modification. RULE: Strict liability applies if product unreasonably dangerous; comparative fault reduces damages based on plaintiff's negligence. APPLICATION: Removing safety device = substantial modification and assumption of risk; clear warnings provided; Joe's conduct was primary cause. CONCLUSION: XYZ likely not liable due to substantial user modification. COUNTERFACTUAL: If guard had been defective and caused removal, XYZ could be liable. POLICY: Encouraging safety compliance requires rejecting claims from willful safety violators.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "tort_law"
        },
        "extra_info": {
        "split": "train",
        "index": 156,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Greenman v. Yuba Power Products, 59 Cal. 2d 57 (1963)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this corporate law issue: ABC Corp's board of directors voted 6-3 to reject a $50 per share buyout offer, despite the stock trading at $35. The three dissenting directors argued the offer was fair and should be presented to shareholders. The board cited a strategic plan showing potential $60 value in 3 years. Minority shareholders sue, claiming breach of fiduciary duty. The board hired the same investment bank that advised on the strategic plan to provide a fairness opinion opposing the deal."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) $50/share buyout offer vs. $35 current price, (2) Board rejected 6-3 without shareholder vote, (3) Strategic plan projects $60 in 3 years, (4) Same investment bank advised on plan and fairness opinion, (5) Minority shareholders sue for breach of fiduciary duty. ISSUE: Whether board's rejection satisfies business judgment rule or constitutes breach of duty. RULE: Business judgment rule protects informed, disinterested decisions; conflicts of interest trigger enhanced scrutiny. APPLICATION: Premium offer suggests economic benefit; same advisor for plan and opinion creates potential conflict; failure to present to shareholders questionable. CONCLUSION: Enhanced scrutiny likely applied due to advisor conflict; board may lose business judgment protection. CRITICAL FACT: If advisor conflict proven material, board decision vulnerable. ALTERNATIVE: If strategic plan genuinely supports higher value, board protected.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "Delaware",
        "legal_domain": "corporate_law"
        },
        "extra_info": {
        "split": "train",
        "index": 157,
        "difficulty": "advanced",
        "court_level": "state_supreme",
        "case_citations": ["Revlon, Inc. v. MacAndrews & Forbes Holdings, 506 A.2d 173 (Del. 1986)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this criminal law analysis: During a traffic stop for speeding, Officer Smith noticed the driver acting nervously and requested consent to search the vehicle. The driver, Mike, initially refused but eventually agreed after Officer Smith said 'We can do this the easy way or the hard way.' The search revealed marijuana in the glove compartment. Mike's attorney argues the consent was coerced and moves to suppress the evidence. The prosecution claims Mike voluntarily consented."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Traffic stop for speeding violation, (2) Mike appeared nervous, (3) Initial refusal to consent to search, (4) Officer's 'easy way or hard way' statement, (5) Mike subsequently consented, (6) Marijuana found in glove compartment. ISSUE: Whether consent was voluntary under Fourth Amendment totality of circumstances test. RULE: Consent must be freely given; coercion invalidates consent; totality of circumstances determines voluntariness. APPLICATION: Initial refusal shows awareness of right; 'hard way' statement implies threat of adverse consequences; ultimate consent followed implied coercion. CONCLUSION: Consent likely involuntary; evidence suppressed. CRITICAL REASONING: 'Hard way' suggests detention/arrest threat, transforming consent into product of coercion. COUNTERFACTUAL: Without threatening language, consent after initial refusal could be voluntary.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "criminal_law"
        },
        "extra_info": {
        "split": "train",
        "index": 158,
        "difficulty": "intermediate",
        "court_level": "federal_district",
        "case_citations": ["Schneckloth v. Bustamonte, 412 U.S. 218 (1973)", "Florida v. Bostick, 501 U.S. 429 (1991)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this family law case: During divorce proceedings, both parents seek custody of their 8-year-old daughter Emma. Father Mark earns $150,000 annually as an engineer and owns a 4-bedroom home. Mother Lisa earns $45,000 as a teacher, rents a 2-bedroom apartment, but has been Emma's primary caregiver since birth. Emma expresses preference to live with her mother. Mark argues his superior financial resources better serve Emma's interests. The court must determine custody arrangement."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Both parents seek custody of 8-year-old Emma, (2) Mark: $150k income, 4-bedroom home, (3) Lisa: $45k income, 2-bedroom rental, (4) Lisa was primary caregiver since birth, (5) Emma prefers living with mother, (6) Mark argues financial superiority. ISSUE: Whether primary caregiver status and child preference outweigh financial advantages. RULE: Best interests of child standard considers emotional bonds, stability, child's preference, and financial capability. APPLICATION: Primary caregiver relationship and child's expressed preference indicate stronger emotional bonds; adequate financial resources not requiring wealth; continuity of care favors status quo. CONCLUSION: Custody likely awarded to Lisa with Mark receiving visitation and child support obligation. FACT ANALYSIS: If Mark had been equal caregiver, financial advantage might tip scales; Emma's age makes preference more significant than if younger.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "family_law"
        },
        "extra_info": {
        "split": "train",
        "index": 159,
        "difficulty": "intermediate",
        "court_level": "state_family",
        "case_citations": ["Uniform Marriage and Divorce Act § 402"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this intellectual property dispute: TechStart developed a revolutionary smartphone app using artificial intelligence. Three months later, MegaCorp released a nearly identical app with the same core functionality and user interface design. TechStart had filed for a patent that is still pending and claims trade secret protection for their AI algorithm. MegaCorp hired two former TechStart engineers just before development. TechStart seeks injunctive relief for trade secret misappropriation and patent infringement."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) TechStart developed AI smartphone app, (2) MegaCorp released similar app 3 months later, (3) Nearly identical functionality and UI design, (4) TechStart has pending patent application, (5) MegaCorp hired 2 former TechStart engineers, (6) TechStart claims trade secret protection for AI algorithm. ISSUE: Whether trade secret misappropriation occurred and preliminary injunction appropriate. RULE: Trade secret protection requires: (1) information has economic value, (2) reasonable secrecy efforts, (3) misappropriation through improper means. APPLICATION: AI algorithm likely has economic value; hiring former employees creates inference of misappropriation; identical functionality suggests access to confidential information; pending patent doesn't bar trade secret claims. CONCLUSION: Strong prima facie case for misappropriation; preliminary injunction likely granted. CRITICAL ANALYSIS: Former employee knowledge + identical features = compelling misappropriation evidence. COUNTERFACTUAL: If engineers developed app through independent means, no misappropriation.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "intellectual_property"
        },
        "extra_info": {
        "split": "train",
        "index": 160,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Uniform Trade Secrets Act", "E.I. duPont deNemours & Co. v. Christopher, 431 F.2d 1012 (5th Cir. 1970)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this environmental law case: ChemCorp operated a manufacturing plant for 40 years, legally disposing of waste under permits valid at the time. New EPA regulations now classify the disposal site as contaminated, requiring $50 million cleanup. Nearby residents developed higher cancer rates, filing suit. ChemCorp argues they complied with all applicable laws and regulations when disposal occurred. The EPA seeks to hold ChemCorp liable under CERCLA for current cleanup costs."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) ChemCorp operated plant 40 years with legal permits, (2) Waste disposal complied with then-current regulations, (3) New EPA rules classify site as contaminated, (4) $50 million cleanup required, (5) Residents show higher cancer rates, (6) EPA seeks CERCLA liability. ISSUE: Whether CERCLA imposes retroactive strict liability despite prior legal compliance. RULE: CERCLA imposes strict liability on potentially responsible parties regardless of fault or legal compliance at time of disposal. APPLICATION: ChemCorp qualifies as potentially responsible party as past owner/operator; strict liability applies even with historical permit compliance; CERCLA specifically rejects due care defense for legal disposal. CONCLUSION: ChemCorp liable for cleanup costs under CERCLA despite prior compliance. POLICY TENSION: Retroactive liability vs. reasonable reliance on legal permits. COUNTERFACTUAL: If disposal occurred after CERCLA enactment, liability clearer; pre-CERCLA disposal still triggers liability due to statute's retroactive effect.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "environmental_law"
        },
        "extra_info": {
        "split": "train",
        "index": 161,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["United States v. Chem-Dyne Corp., 572 F. Supp. 802 (S.D. Ohio 1983)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this civil procedure case: PlaintiffCorp filed a federal diversity lawsuit against DefendantCorp in New York. DefendantCorp moved to dismiss for lack of personal jurisdiction, arguing it has no offices in New York, no employees there, and only sold products through an independent distributor. PlaintiffCorp argues DefendantCorp's website accepted orders from New York customers and shipped products there, establishing minimum contacts. DefendantCorp's website had targeted New York-specific advertising and accepted payment in New York."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Federal diversity lawsuit filed in New York, (2) DefendantCorp has no NY offices/employees, (3) Sales through independent distributor only, (4) Website accepts NY orders and ships products, (5) NY-targeted advertising on website, (6) Payment processing in NY. ISSUE: Whether DefendantCorp's internet activities establish minimum contacts for personal jurisdiction. RULE: Specific jurisdiction requires purposeful availment of forum benefits and reasonableness; internet activities may establish contacts if interactive and targeting forum state. APPLICATION: NY-targeted advertising shows purposeful availment; order acceptance and payment processing demonstrate interactive website; product shipment into forum creates contacts; claims arise from NY activities. CONCLUSION: Personal jurisdiction likely established through internet contacts and targeted commercial activity. MODERN DOCTRINE: Internet jurisdiction requires more than passive website; active targeting and commercial interaction satisfy minimum contacts. ALTERNATIVE: If website merely passive, jurisdiction likely fails.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "civil_procedure"
        },
        "extra_info": {
        "split": "train",
        "index": 162,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["International Shoe Co. v. Washington, 326 U.S. 310 (1945)", "Zippo Mfg. Co. v. Zippo Dot Com, Inc., 952 F. Supp. 1119 (W.D. Pa. 1997)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this evidence law issue: In a murder trial, the prosecution seeks to introduce defendant's confession made during police interrogation. Defense argues the confession is inadmissible because defendant wasn't read Miranda rights until after he made incriminating statements. The interrogation began as a traffic stop, escalated when officers smelled alcohol, and defendant was handcuffed before questioning about an unrelated murder occurred. Defendant was not formally arrested until after the confession."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Murder trial confession evidence, (2) No Miranda warning before incriminating statements, (3) Began as traffic stop, escalated due to alcohol smell, (4) Defendant handcuffed during questioning, (5) Questions about unrelated murder, (6) Formal arrest occurred after confession. ISSUE: Whether custodial interrogation occurred requiring Miranda warnings. RULE: Miranda applies when suspect is in custody and subjected to interrogation; custody determined by reasonable person standard considering restraint and freedom to leave. APPLICATION: Handcuffing suggests custody despite no formal arrest; shift from traffic stop to murder questioning shows interrogation; reasonable person would not feel free to leave when handcuffed; unrelated murder questions exceed traffic stop scope. CONCLUSION: Custodial interrogation occurred; confession inadmissible without proper Miranda warnings. CRITICAL TRANSITION: Traffic stop became custodial when restraints applied and focus shifted to unrelated crime. COUNTERFACTUAL: If questioning remained about traffic violations, Miranda might not apply.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "criminal_law"
        },
        "extra_info": {
        "split": "train",
        "index": 163,
        "difficulty": "advanced",
        "court_level": "state_appellate",
        "case_citations": ["Miranda v. Arizona, 384 U.S. 436 (1966)", "Berkemer v. McCarty, 468 U.S. 420 (1984)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this insurance law dispute: Homeowner filed a claim after a kitchen fire caused $75,000 in damage. The insurance policy excludes coverage for fires caused by 'faulty electrical work performed without permits.' Investigation revealed the homeowner's brother, an unlicensed electrician, had installed new outlets one month before the fire. The brother claims he used proper techniques, but no permit was obtained. The insurance company denies the claim based on the exclusion clause."
        }
        ],
        "redirect_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Kitchen fire caused $75,000 damage, (2) Policy excludes 'faulty electrical work without permits', (3) Unlicensed brother installed outlets one month prior, (4) Brother claims proper technique used, (5) No permit obtained, (6) Insurance denies claim. ISSUE: Whether policy exclusion applies when electrical work was unpermitted but allegedly proper. RULE: Insurance exclusions are construed narrowly against insurer; burden on insurer to prove exclusion applies; ambiguous terms resolved in favor of insured. APPLICATION: Two-part exclusion requires both 'faulty' work AND no permit; undisputed no permit obtained; dispute over whether work was 'faulty' vs. proper technique; insurer must prove causal connection between unpermitted work and fire. CONCLUSION: If fire causation unclear, exclusion may not apply; insurer bears burden of proving both faulty work and no permit caused loss. INTERPRETATION: 'Faulty' suggests defective workmanship, not mere permit violation. OUTCOME DEPENDS: Fire investigation results regarding electrical cause.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "insurance_law"
        },
        "extra_info": {
        "split": "train",
        "index": 164,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Darner Motor Sales v. Universal Underwriters Ins., 682 P.2d 388 (Ariz. 1984)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this administrative law challenge: The Department of Labor issued new regulations requiring all construction companies to provide additional safety training, increasing compliance costs by an average of $50,000 per company. BuildCorp challenges the regulation, arguing the agency exceeded its statutory authority and failed to consider economic impacts as required by the enabling statute. The DOL argues worker safety justifies the costs and falls within their broad rulemaking authority."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) DOL issued new safety training regulations, (2) $50,000 average compliance cost per company, (3) BuildCorp challenges agency authority, (4) Claims economic impact analysis required but not done, (5) DOL cites worker safety justification, (6) DOL claims broad rulemaking authority. ISSUE: Whether DOL exceeded statutory authority and failed mandatory cost-benefit analysis. RULE: Agencies must act within statutory bounds; if statute requires economic analysis, failure to comply invalidates rule; Chevron deference applies to reasonable agency interpretation. APPLICATION: Must examine enabling statute language for economic analysis requirement; if mandatory, agency failure requires invalidation; safety justification alone insufficient if economic analysis statutorily required; broad authority doesn't override specific procedural requirements. CONCLUSION: If statute mandates cost-benefit analysis, regulation invalid despite safety benefits. CHEVRON ANALYSIS: Agency interpretation reasonable only if within statutory constraints. REMEDIAL OPTION: Remand to agency for proper cost-benefit analysis rather than permanent invalidation.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "administrative_law"
        },
        "extra_info": {
        "split": "train",
        "index": 165,
        "difficulty": "advanced",
        "court_level": "federal_circuit",
        "case_citations": ["Chevron U.S.A. Inc. v. Natural Resources Defense Council, 467 U.S. 837 (1984)", "Motor Vehicle Mfrs. Ass'n v. State Farm, 463 U.S. 29 (1983)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this securities law violation: TechCorp CEO posted on social media 'Amazing quarter ahead! Investors will be very happy with our numbers!' Three weeks before earnings announcement, the CEO knew quarterly revenue would miss analyst expectations by 30%. The stock price rose 15% following the social media post. After poor earnings were announced, shareholders filed a securities fraud lawsuit claiming the CEO's statement was materially misleading."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) CEO posted optimistic social media message, (2) Posted 3 weeks before earnings, (3) CEO knew revenue would miss expectations by 30%, (4) Stock rose 15% after post, (5) Poor earnings later announced, (6) Shareholders sue for securities fraud. ISSUE: Whether CEO's social media statement constitutes securities fraud under Rule 10b-5. RULE: Securities fraud requires: (1) material misstatement or omission, (2) scienter (intent/recklessness), (3) reliance, (4) causation, (5) damages. APPLICATION: 'Amazing quarter' statement materially misleading given CEO's contrary knowledge; scienter established by CEO's actual knowledge of poor performance; market price movement shows reliance; stock drop after earnings shows causation and damages. CONCLUSION: Strong securities fraud case; CEO's social media statement violated antifraud provisions. MATERIALITY ANALYSIS: 30% revenue miss would be material to reasonable investor; optimistic language directly contradicts known facts. MODERN ISSUE: Social media posts increasingly treated as formal corporate communications subject to securities laws.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "securities_law"
        },
        "extra_info": {
        "split": "train",
        "index": 166,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Basic Inc. v. Levinson, 485 U.S. 224 (1988)", "TSC Industries v. Northway, 426 U.S. 438 (1976)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this bankruptcy law issue: RetailCorp filed Chapter 11 bankruptcy with $500 million in debts. The company's reorganization plan proposes paying unsecured creditors 30 cents on the dollar over 5 years while retaining current management and continuing operations. LandlordCorp, holding a $50 million lease claim, objects that the plan unfairly discriminates against landlords compared to trade creditors who receive better treatment. The plan also releases management from potential liability for pre-bankruptcy decisions."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Chapter 11 filing with $500M debt, (2) Plan pays unsecured creditors 30% over 5 years, (3) Current management retained, (4) LandlordCorp has $50M lease claim, (5) Landlords claim discrimination vs. trade creditors, (6) Plan releases management liability. ISSUE: Whether reorganization plan meets confirmation requirements for fair treatment and proper releases. RULE: Chapter 11 plans must not discriminate unfairly among similar creditor classes; third-party releases require special justification and affected party consent or sufficient consideration. APPLICATION: If landlords and trade creditors in same class, differential treatment may violate unfair discrimination test; management releases require essential contribution to reorganization or creditor consent; 30% recovery rate must be reasonable based on liquidation alternative. CONCLUSION: Plan confirmation depends on creditor classification analysis and release justification; unfair discrimination claim has merit if similar creditors treated differently. CRITICAL ANALYSIS: Management retention vs. release creates tension - if management essential, releases more justified.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "bankruptcy_law"
        },
        "extra_info": {
        "split": "train",
        "index": 167,
        "difficulty": "advanced",
        "court_level": "federal_bankruptcy",
        "case_citations": ["11 U.S.C. § 1129", "In re Metromedia Fiber Network, 416 F.3d 136 (2d Cir. 2005)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this immigration law case: Maria, a Mexican citizen, entered the U.S. illegally 15 years ago as a child with her parents. She graduated from a U.S. high school and college, has no criminal record, and works as a nurse. ICE initiated removal proceedings, but Maria applies for cancellation of removal, claiming exceptional hardship to her U.S. citizen children aged 5 and 7. Her husband is also undocumented and would likely be deported if Maria is removed."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Maria illegally entered 15 years ago as child, (2) U.S. education and nursing career, (3) No criminal record, (4) ICE removal proceedings initiated, (5) U.S. citizen children aged 5 and 7, (6) Undocumented husband facing deportation. ISSUE: Whether Maria qualifies for cancellation of removal based on exceptional hardship. RULE: Cancellation requires: (1) 10+ years continuous presence, (2) good moral character, (3) exceptional hardship to U.S. citizen relative if removed. APPLICATION: 15-year presence satisfies continuity; nursing career and no criminal record establish good character; removal would separate young U.S. children from both parents, creating educational, emotional, and financial hardship. CONCLUSION: Strong case for cancellation due to exceptional hardship to citizen children. HARDSHIP ANALYSIS: Loss of both parents (Maria + husband's likely deportation) exceeds normal hardship; children's young age and U.S. citizenship strengthen claim. COUNTERFACTUAL: If only one parent faced removal or children were older, hardship claim weaker.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "immigration_law"
        },
        "extra_info": {
        "split": "train",
        "index": 168,
        "difficulty": "intermediate",
        "court_level": "immigration_court",
        "case_citations": ["8 U.S.C. § 1229b(b)(1)", "In re Recinas, 23 I&N Dec. 467 (BIA 2002)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this tax law dispute: TechCorp claimed a $10 million deduction for research and development expenses in software development. The IRS challenges the deduction, arguing the expenses were for routine software updates rather than qualifying research under IRC Section 174. TechCorp maintains the work involved developing revolutionary AI algorithms with uncertain success. The company documented experimental processes and multiple failed approaches before achieving breakthrough technology."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) TechCorp claimed $10M R&D deduction, (2) Software development expenses for AI algorithms, (3) IRS claims routine updates, not qualifying research, (4) TechCorp claims revolutionary technology with uncertain outcome, (5) Documented experimental processes and failures, (6) Breakthrough technology ultimately achieved. ISSUE: Whether AI algorithm development qualifies as research under IRC Section 174. RULE: Section 174 covers research if: (1) undertaken to discover technological information, (2) for developing new/improved business component, (3) substantially all activities constitute experimentation. APPLICATION: AI algorithm development involves technological information discovery; new business component clearly intended; documented failures and experimental processes show experimentation; uncertainty of success supports research characterization vs. routine development. CONCLUSION: Likely qualifies for R&D deduction if experimental nature sufficiently documented. CRITICAL DISTINCTION: Revolutionary vs. routine development - breakthrough nature and documented experimentation support research classification. DOCUMENTATION KEY: Failure records and experimental methodology crucial for sustaining deduction.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "tax_law"
        },
        "extra_info": {
        "split": "train",
        "index": 169,
        "difficulty": "advanced",
        "court_level": "tax_court",
        "case_citations": ["IRC § 174", "Treas. Reg. § 1.174-2"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this antitrust analysis: MegaRetail, controlling 40% of the grocery market, announced exclusive supply agreements with three major food distributors representing 60% of regional distribution. The agreements prevent distributors from supplying MegaRetail's competitors for 5 years. Small competitor GroceryCorp claims this constitutes illegal tying and exclusive dealing that will force them out of business. MegaRetail argues the agreements provide efficiency benefits and competitive pricing for consumers."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) MegaRetail has 40% grocery market share, (2) Exclusive 5-year agreements with distributors controlling 60% of distribution, (3) Agreements prevent distributor supply to competitors, (4) GroceryCorp claims forced exit from market, (5) MegaRetail cites efficiency and pricing benefits. ISSUE: Whether exclusive dealing agreements violate Section 1 Sherman Act or Section 3 Clayton Act. RULE: Exclusive dealing illegal if: (1) substantial foreclosure of competition, (2) probable anticompetitive effects, (3) no sufficient procompetitive justifications. APPLICATION: 60% distribution foreclosure is substantial; competitors lose access to majority of supply channels; 40% market share + distribution control creates barrier to entry; efficiency claims must be balanced against competitive harm. CONCLUSION: Likely antitrust violation due to substantial foreclosure and market power combination. ECONOMIC ANALYSIS: 40% retail share + 60% distribution control = significant competitive concern; 5-year term increases anticompetitive impact. EFFICIENCY DEFENSE: Cost savings alone insufficient if substantial lessening of competition proven.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "antitrust_law"
        },
        "extra_info": {
        "split": "train",
        "index": 170,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Tampa Electric Co. v. Nashville Coal Co., 365 U.S. 320 (1961)", "Jefferson Parish v. Hyde, 466 U.S. 2 (1984)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this landlord-tenant dispute: Tenant Sarah has lived in an apartment for 3 years under a month-to-month lease. Landlord Jim gave 30-day notice to terminate tenancy, stating he needs the unit for his elderly mother. Local rent control ordinance requires 'just cause' for eviction and lists 'owner move-in' as valid cause. Sarah discovers Jim already moved his mother into a different unit he owns nearby and argues the notice was pretextual to circumvent rent control protections."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Sarah: 3-year month-to-month tenant, (2) Jim gave 30-day termination notice, (3) Stated reason: elderly mother needs unit, (4) Rent control requires 'just cause' for eviction, (5) 'Owner move-in' listed as valid cause, (6) Mother already housed in Jim's other unit nearby. ISSUE: Whether landlord's stated reason constitutes valid 'just cause' when alternative housing was available. RULE: Rent control 'just cause' requirements strictly construed; pretextual reasons invalid; owner move-in must be made in good faith with genuine need. APPLICATION: Mother's prior placement in alternative unit suggests pretextual motive; good faith requirement not met if genuine need lacking; strict construction favors tenant protection. CONCLUSION: Eviction notice likely invalid due to pretext and lack of genuine need. GOOD FAITH TEST: Alternative housing availability undermines necessity claim. POLICY: Rent control protections meaningless if easily circumvented through pretextual reasons.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "California",
        "legal_domain": "property_law"
        },
        "extra_info": {
        "split": "train",
        "index": 171,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Ellis v. City of Berkeley, 11 Cal. 3d 865 (1974)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this defamation case: Local newspaper published an article stating 'Mayor Johnson misappropriated city funds for personal use' based on anonymous source claiming to have seen financial records. The article caused public outcry and Johnson lost reelection. Investigation later revealed the claims were false - the anonymous source was Johnson's political rival who fabricated the allegations. Johnson sues for defamation, claiming the newspaper failed to verify the story adequately."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Newspaper published misappropriation allegation against Mayor Johnson, (2) Based on anonymous source claiming document access, (3) Public outcry and reelection loss resulted, (4) Claims later proven false, (5) Source was political rival who fabricated allegations, (6) Johnson sues for inadequate verification. ISSUE: Whether newspaper's publication constitutes defamation under actual malice standard for public officials. RULE: Public officials must prove: (1) false statement of fact, (2) publication, (3) actual malice (knowledge of falsity or reckless disregard for truth), (4) damages. APPLICATION: False factual allegation clearly published with resulting damages; actual malice shown by failure to verify anonymous accusations against political figure; reasonable journalist would investigate before publishing serious corruption allegations. CONCLUSION: Strong defamation case due to reckless disregard for truth in failing to verify serious allegations. ACTUAL MALICE: Anonymous source + serious allegation + no verification = reckless disregard. PRESS FREEDOM TENSION: Must balance First Amendment against accountability for false reporting.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "tort_law"
        },
        "extra_info": {
        "split": "train",
        "index": 172,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["New York Times Co. v. Sullivan, 376 U.S. 254 (1964)", "St. Amant v. Thompson, 390 U.S. 727 (1968)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this medical malpractice analysis: Dr. Martinez performed gallbladder surgery on patient Williams using a minimally invasive laparoscopic procedure. During surgery, Dr. Martinez accidentally cut the common bile duct, a known but rare complication occurring in 0.3% of cases. Williams developed serious complications requiring additional surgery and prolonged recovery. Williams sues for malpractice, claiming Dr. Martinez was negligent. Dr. Martinez argues the injury was an inherent risk of the procedure that was properly disclosed."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Dr. Martinez performed laparoscopic gallbladder surgery, (2) Accidentally cut common bile duct during procedure, (3) Complication occurs in 0.3% of cases, (4) Williams suffered serious complications requiring additional surgery, (5) Williams sues for negligence, (6) Dr. Martinez claims inherent disclosed risk. ISSUE: Whether accidental bile duct injury constitutes negligence or acceptable complication risk. RULE: Medical malpractice requires breach of standard of care; known complications don't automatically constitute negligence if procedure performed competently; informed consent may limit liability for disclosed risks. APPLICATION: 0.3% complication rate suggests rare but known risk; must examine whether Dr. Martinez's technique met professional standards; proper informed consent may bar recovery for inherent risks; need expert testimony on standard of care. CONCLUSION: Outcome depends on technique analysis and informed consent adequacy; rare complication alone insufficient for malpractice. STANDARD OF CARE: Expert testimony crucial for determining whether technique was negligent vs. unfortunate outcome of acceptable risk.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "tort_law"
        },
        "extra_info": {
        "split": "train",
        "index": 173,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Canterbury v. Spence, 464 F.2d 772 (D.C. Cir. 1972)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this consumer protection case: AutoDealer sold customer Janet a 'certified pre-owned' vehicle for $25,000, representing it had passed comprehensive 150-point inspection and had no accident history. Three months later, Janet discovered the car had been in a major flood and the dealer had altered VIN records to hide the damage. The dealer's certification was based solely on superficial appearance, not actual mechanical inspection. Janet seeks rescission and damages under state consumer fraud statutes."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) AutoDealer sold 'certified pre-owned' car for $25,000, (2) Represented 150-point inspection and no accident history, (3) Car was actually flood-damaged, (4) Dealer altered VIN records to conceal damage, (5) 'Certification' based only on appearance, not inspection, (6) Janet seeks rescission and damages. ISSUE: Whether dealer's misrepresentations constitute consumer fraud warranting rescission and damages. RULE: Consumer fraud requires: (1) material misrepresentation, (2) intent to deceive or reckless disregard for truth, (3) justifiable reliance, (4) resulting damages. APPLICATION: Flood damage + accident history misrepresentations are material; VIN alteration shows intent to deceive; 'certified pre-owned' label justifies reliance on inspection claims; $25,000 purchase price establishes damages. CONCLUSION: Clear consumer fraud case; Janet entitled to rescission and damages. MATERIALITY: Flood damage fundamentally affects vehicle value and safety. VIN ALTERATION: Demonstrates intentional concealment beyond mere negligence. RESCISSION: Fraud voids contract, allowing full restitution.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "consumer_law"
        },
        "extra_info": {
        "split": "train",
        "index": 174,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Uniform Deceptive Trade Practices Act"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this probate dispute: Wealthy businessman Robert died leaving a will naming his daughter Lisa as sole beneficiary of his $5 million estate. Robert's brother Mike contests the will, claiming Robert lacked testamentary capacity due to Alzheimer's disease and that Lisa unduly influenced him. The will was executed 6 months before Robert's death. Medical records show mild cognitive impairment, but Robert's attorney testified he seemed lucid during will preparation and understood the document's effects."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Robert died with $5M estate, (2) Will names daughter Lisa sole beneficiary, (3) Brother Mike contests for lack of capacity and undue influence, (4) Robert had Alzheimer's/mild cognitive impairment, (5) Will executed 6 months before death, (6) Attorney testified Robert seemed lucid and understood will effects. ISSUE: Whether Robert had sufficient testamentary capacity and was free from undue influence when executing will. RULE: Testamentary capacity requires understanding: (1) nature of act, (2) extent of property, (3) natural objects of bounty, (4) effect of disposition; undue influence requires domination overcoming free will. APPLICATION: Mild impairment doesn't automatically void capacity; attorney's contemporaneous observations support competency; daughter as sole beneficiary raises undue influence suspicion but natural for parent-child relationship; timing 6 months before death relevant to capacity assessment. CONCLUSION: Will likely valid absent stronger evidence of incapacity or coercion; mild impairment insufficient to overcome attorney testimony. BURDEN: Contestant must prove lack of capacity by clear evidence; natural inheritance pattern supports validity.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "probate_law"
        },
        "extra_info": {
        "split": "train",
        "index": 175,
        "difficulty": "intermediate",
        "court_level": "probate_court",
        "case_citations": ["In re Estate of Lakatosh, 656 A.2d 1378 (Pa. 1995)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this education law case: Public school district implemented a policy requiring students to stand for the Pledge of Allegiance, with suspension for refusal. Student Maria, whose religious beliefs prohibit pledging to symbols, refused to stand and was suspended for 3 days. Her parents sue, claiming the policy violates First Amendment religious exercise and speech protections. The district argues maintaining patriotic unity is a compelling government interest."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) School policy mandates standing for Pledge with suspension penalty, (2) Maria refused based on religious beliefs against symbol worship, (3) 3-day suspension imposed, (4) Parents sue claiming First Amendment violations, (5) District cites patriotic unity as compelling interest. ISSUE: Whether mandatory pledge participation violates First Amendment religious exercise and speech protections. RULE: First Amendment protects both religious exercise and freedom from compelled speech; government must show compelling interest and narrow tailoring for restrictions on fundamental rights. APPLICATION: Forced pledge participation constitutes compelled speech; religious objection triggers free exercise protection; suspension penalty creates substantial burden; patriotic education can be achieved through less restrictive means than mandatory participation. CONCLUSION: Policy likely unconstitutional as violation of both speech and religious exercise rights. PRECEDENT: West Virginia v. Barnette prohibits mandatory flag salute; religious objection strengthens constitutional protection. ALTERNATIVE MEANS: Patriotic education possible without coercing individual participation.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "constitutional_law"
        },
        "extra_info": {
        "split": "train",
        "index": 176,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["West Virginia State Board of Education v. Barnette, 319 U.S. 624 (1943)", "Wooley v. Maynard, 430 U.S. 705 (1977)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this partnership dissolution case: Law firm partners Adams and Baker practiced together for 15 years under oral partnership agreement. When Baker decided to retire, they disagreed over client list ownership and profit-sharing for ongoing cases. Adams claims exclusive rights to clients he brought to the firm, while Baker demands equal share of all profits from cases initiated during partnership. No written agreement exists regarding post-dissolution rights and responsibilities."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Adams and Baker: 15-year law partnership, (2) Oral partnership agreement only, (3) Baker retiring, Adams continuing, (4) Dispute over client list ownership, (5) Disagreement on profit-sharing for ongoing cases, (6) No written post-dissolution provisions. ISSUE: How to allocate client relationships and ongoing case profits upon partnership dissolution without written agreement. RULE: Partnership dissolution governed by state partnership law; absent agreement, partners generally entitled to equal shares; client relationships typically follow attorney who served client; ongoing cases may require fee-splitting based on work performed. APPLICATION: Oral partnership suggests equal profit-sharing intent; clients have right to choose counsel upon dissolution; ongoing cases require analysis of each partner's contribution; 15-year equal partnership presumption favors equal treatment absent client-specific relationships. CONCLUSION: Equal division of ongoing case profits likely required; clients choose counsel going forward; case-by-case analysis needed for fee allocation. DEFAULT RULES: Partnership law provides gap-filling when agreements incomplete; equal treatment presumption in equal partnerships.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "partnership_law"
        },
        "extra_info": {
        "split": "train",
        "index": 177,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Uniform Partnership Act § 801"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this zoning law challenge: City Council rezoned a residential neighborhood to allow commercial development after receiving a proposal from MegaMart for a new store. Residents sue, claiming the rezoning was 'spot zoning' that violated comprehensive planning requirements and was influenced by improper ex parte communications between MegaMart executives and council members. The city argues economic development benefits justify the zoning change."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) City rezoned residential area for commercial use, (2) MegaMart proposed new store triggering rezoning, (3) Residents claim illegal 'spot zoning', (4) Allegations of improper ex parte communications, (5) Challenge based on comprehensive planning violations, (6) City cites economic development justification. ISSUE: Whether selective rezoning constitutes illegal spot zoning and whether procedural violations invalidate the decision. RULE: Zoning changes must conform to comprehensive plan and serve public interest; spot zoning illegal when benefiting single property owner without broader planning justification; ex parte communications may violate due process in quasi-judicial proceedings. APPLICATION: Single-property rezoning for specific developer suggests spot zoning; ex parte communications undermine fair hearing process; economic benefits alone insufficient without comprehensive plan consistency; residents denied fair consideration of opposition. CONCLUSION: Rezoning likely invalid due to spot zoning characteristics and procedural violations. SPOT ZONING TEST: Benefits specific property without broader planning rationale; procedural fairness essential in zoning decisions.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "municipal_law"
        },
        "extra_info": {
        "split": "train",
        "index": 178,
        "difficulty": "advanced",
        "court_level": "state_appellate",
        "case_citations": ["Village of Euclid v. Ambler Realty Co., 272 U.S. 365 (1926)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this water rights dispute: Farmer Johnson has used stream water to irrigate his crops for 25 years under prior appropriation doctrine. UrbanCity, located downstream, claims Johnson's water use violates their senior water rights dating to 1890. Johnson argues his use has been continuous and beneficial, establishing his own appropriation rights. During recent drought, Johnson's usage left insufficient water for city residents, causing water restrictions."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Johnson irrigated crops for 25 years using stream water, (2) UrbanCity claims senior rights from 1890, (3) Johnson claims continuous beneficial use establishing rights, (4) Recent drought created water shortage, (5) Johnson's use left insufficient water for city, (6) City implemented water restrictions. ISSUE: Whether Johnson's 25-year use establishes water rights superior to city's 1890 claim under prior appropriation doctrine. RULE: Prior appropriation follows 'first in time, first in right' principle; appropriation requires: (1) intent to appropriate, (2) diversion from source, (3) beneficial use, (4) continuous use without abandonment. APPLICATION: City's 1890 rights predate Johnson's use; Johnson must prove elements of valid appropriation starting 25 years ago; continuous irrigation suggests beneficial use; drought conditions test reasonableness of both uses; senior rights generally prevail over junior appropriators. CONCLUSION: City's senior rights likely superior; Johnson may have valid junior rights subject to senior appropriator needs. PRIORITY SYSTEM: Earlier appropriation typically trumps later use; drought may require curtailment of junior rights to satisfy senior appropriators.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "water_law"
        },
        "extra_info": {
        "split": "train",
        "index": 179,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Coffin v. Left Hand Ditch Co., 6 Colo. 443 (1882)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this elder law case: 85-year-old Dorothy transferred her $800,000 home to grandson Paul for $1 consideration, claiming it was a gift to avoid nursing home costs. Three years later, Dorothy requires long-term care and applies for Medicaid. The state seeks to recover the home's value as an improper asset transfer made to qualify for benefits. Paul argues the transfer was a legitimate gift motivated by family affection, not Medicaid planning."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Dorothy transferred $800k home to grandson Paul for $1, (2) Stated purpose: gift to avoid nursing home costs, (3) Transfer occurred 3 years before Medicaid application, (4) Dorothy now needs long-term care, (5) State claims improper asset transfer for benefit qualification, (6) Paul claims legitimate family gift. ISSUE: Whether home transfer constitutes Medicaid fraud or permissible gift under lookback period rules. RULE: Medicaid prohibits asset transfers for inadequate consideration within 5-year lookback period; transfers create penalty period based on value transferred; burden on applicant to prove transfer not for Medicaid qualification purposes. APPLICATION: $800k home for $1 clearly inadequate consideration; 3-year timeframe within 5-year lookback; Dorothy's stated motive (avoiding nursing costs) indicates Medicaid planning purpose; penalty period calculated from improper transfer value. CONCLUSION: Transfer likely creates Medicaid penalty period; inadequate consideration and stated motive establish improper transfer. MEDICAID PLANNING: Legitimate planning requires transfers outside lookback period or adequate consideration; family affection alone insufficient defense when Medicaid avoidance motive proven.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "elder_law"
        },
        "extra_info": {
        "split": "train",
        "index": 180,
        "difficulty": "intermediate",
        "court_level": "administrative",
        "case_citations": ["42 U.S.C. § 1396p", "Deficit Reduction Act of 2005"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this construction defect case: Homeowner Kim hired BuildCorp to construct a custom home for $500,000. Two years after completion, Kim discovered significant foundation problems causing structural damage throughout the house. Expert analysis revealed BuildCorp used substandard concrete and failed to follow approved engineering specifications. Repair costs are estimated at $200,000. BuildCorp claims the problems result from unusual soil conditions not disclosed by the owner."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) BuildCorp constructed custom home for Kim at $500k, (2) Foundation problems discovered 2 years post-completion, (3) Expert analysis shows substandard concrete and specification violations, (4) Structural damage throughout house, (5) $200k repair estimate, (6) BuildCorp blames undisclosed soil conditions. ISSUE: Whether BuildCorp's construction defects breach contract warranty obligations and whether soil conditions excuse performance. RULE: Construction contracts include implied warranty of workmanlike performance; contractor liable for defects caused by substandard materials or failure to follow specifications; unforeseen conditions may excuse performance if truly unforeseeable and material. APPLICATION: Substandard concrete and specification violations clearly breach workmanlike performance warranty; contractor responsible for material selection and compliance with plans; soil condition defense requires proof that conditions were unforeseeable and caused defects rather than contractor's material/workmanship failures. CONCLUSION: BuildCorp likely liable for construction defects; soil condition defense unlikely to overcome concrete/specification violations. WORKMANSHIP WARRANTY: Contractor must deliver structure built to professional standards using proper materials; owner's soil disclosure obligations don't excuse contractor's material/workmanship defects.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "construction_law"
        },
        "extra_info": {
        "split": "train",
        "index": 181,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Spearin Doctrine - United States v. Spearin, 248 U.S. 132 (1918)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this franchise law dispute: FastFood Corp terminated franchise agreement with Owner Mike after 15 years, claiming breach of quality standards. Mike argues the termination was retaliatory because he complained about corporate price-fixing with other franchisees. The franchise agreement allows termination for 'material breach' but requires 30-day cure period. FastFood gave only 10 days notice and refused to specify which standards were violated."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) FastFood terminated Mike's 15-year franchise for quality standard violations, (2) Mike claims retaliation for price-fixing complaints, (3) Agreement requires 30-day cure period for material breach, (4) Only 10 days notice provided, (5) FastFood refused to specify violated standards, (6) Termination followed Mike's complaints about other franchisees. ISSUE: Whether franchise termination violated contractual notice requirements and constituted retaliation for protected complaints. RULE: Franchise termination must comply with agreement terms; material breach requires specific notice and opportunity to cure; retaliation for antitrust complaints may violate public policy; franchisor must act in good faith. APPLICATION: 10-day notice violates 30-day contractual requirement; failure to specify violations denies meaningful cure opportunity; timing after price-fixing complaints suggests retaliatory motive; good faith requires specific notice and genuine cure opportunity. CONCLUSION: Termination likely improper due to notice violations and potential retaliation; Mike entitled to reinstatement or damages. PROCEDURAL COMPLIANCE: Franchise agreements strictly construed; franchisor must follow termination procedures exactly. RETALIATION CLAIM: Antitrust complaints may constitute protected activity preventing retaliatory termination.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "franchise_law"
        },
        "extra_info": {
        "split": "train",
        "index": 182,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["FTC Franchise Rule 16 C.F.R. § 436"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this sports law case: Professional basketball player Johnson signed a 5-year contract with TeamCity for $50 million. After 2 years, Johnson suffered a career-ending knee injury during practice. The contract includes disability insurance but contains a clause voiding payment if injury results from 'player's own negligence or reckless conduct.' Team claims Johnson was showboating with an unnecessary dunk attempt when injured and refuses payment under the negligence clause."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Johnson: 5-year, $50M basketball contract, (2) Career-ending knee injury after 2 years during practice, (3) Contract includes disability insurance coverage, (4) Negligence/recklessness clause voids payment, (5) Team claims injury from unnecessary showboating dunk, (6) Team refuses disability payment citing negligence. ISSUE: Whether routine basketball practice activities constitute negligence or recklessness voiding disability coverage. RULE: Contract terms construed against drafter; negligence/recklessness clauses require clear proof of unreasonable conduct; normal athletic activity presumptively not negligent even if risky; ambiguous terms resolved in favor of coverage. APPLICATION: Dunking is routine basketball activity within normal practice scope; 'showboating' during practice not necessarily reckless if within sport's normal parameters; team bears burden proving conduct exceeded reasonable athletic behavior; injury during sanctioned practice activity presumptively covered. CONCLUSION: Team unlikely to void coverage based on routine basketball practice activity; normal athletic risks don't constitute negligence. SPORTS CONTEXT: Athletic activities inherently risky; negligence clauses can't void coverage for normal sport-related risks. BURDEN: Team must prove conduct clearly outside reasonable athletic behavior, not mere aggressive play.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "sports_law"
        },
        "extra_info": {
        "split": "train",
        "index": 183,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["Standard insurance contract interpretation principles"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this aviation law incident: Private pilot Rodriguez crashed his small aircraft into commercial building during emergency landing attempt, killing 3 office workers. Investigation revealed mechanical failure in the engine, but Rodriguez had not completed required annual inspection that might have detected the problem. Families of victims sue Rodriguez and aircraft manufacturer for wrongful death. Rodriguez claims emergency exempted him from normal landing requirements."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Rodriguez crashed aircraft into building during emergency landing, (2) 3 office workers killed, (3) Mechanical engine failure caused emergency, (4) Required annual inspection not completed, (5) Inspection might have detected failure, (6) Families sue Rodriguez and manufacturer. ISSUE: Whether pilot's negligence in maintenance and emergency landing decisions creates liability for wrongful death. RULE: Aircraft operators owe duty of reasonable care to persons on ground; emergency situations may excuse normal procedures but don't eliminate duty of care; negligent maintenance may create liability if proximate cause of harm. APPLICATION: Failed inspection creates negligence if reasonable inspection would have prevented failure; emergency landing site selection must be reasonable under circumstances; duty to ground persons requires minimizing harm even in emergency; proximate causation links maintenance failure to deaths. CONCLUSION: Rodriguez likely liable if inspection failure caused crash; emergency doesn't excuse negligent maintenance or unreasonable landing decisions. EMERGENCY DOCTRINE: Reduces standard for split-second decisions but doesn't excuse pre-emergency negligence in maintenance. MANUFACTURER LIABILITY: Depends on whether defect existed independent of maintenance failure.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "aviation_law"
        },
        "extra_info": {
        "split": "train",
        "index": 184,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Federal Aviation Regulations 14 C.F.R."],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this cybersecurity law case: TechCorp's database containing 500,000 customer credit card numbers was hacked due to failure to install available security patches. Customers suffered identity theft and financial losses totaling $10 million. State attorney general sues TechCorp under consumer protection act for inadequate data security. TechCorp argues no specific security standards exist and they used industry-standard practices at the time of implementation."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) TechCorp database with 500k credit card numbers hacked, (2) Security breach due to uninstalled available patches, (3) Customers suffered $10M in identity theft losses, (4) State sues under consumer protection act, (5) TechCorp claims no specific standards exist, (6) TechCorp used industry standards at implementation time. ISSUE: Whether failure to maintain current security updates constitutes unfair/deceptive practice under consumer protection law. RULE: Consumer protection acts prohibit unfair practices that cause substantial consumer harm; reasonable data security measures required when handling sensitive information; evolving technology requires ongoing security maintenance, not just initial compliance. APPLICATION: 500k records breach creates substantial consumer harm; available patches indicate reasonable security required ongoing updates; industry standards evolve with threat landscape; failure to maintain security despite available fixes suggests unreasonable practice. CONCLUSION: TechCorp likely violated consumer protection law; duty to maintain security extends beyond initial implementation. EVOLVING STANDARDS: Data security requires continuous vigilance; initial compliance insufficient if threats evolve and countermeasures available. REASONABLE CARE: Prudent business would install available security patches to protect customer data.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "cyber_law"
        },
        "extra_info": {
        "split": "train",
        "index": 185,
        "difficulty": "advanced",
        "court_level": "state_trial",
        "case_citations": ["State Consumer Protection Acts", "FTC Act Section 5"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this mining law dispute: MinerCorp has operated an open-pit copper mine under federal lease for 20 years. New EPA regulations require expensive environmental remediation that would cost $200 million, making continued operation unprofitable. MinerCorp claims the regulations constitute a taking requiring just compensation. EPA argues environmental protection is valid police power that doesn't require compensation for regulatory compliance costs."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) MinerCorp operated mine for 20 years under federal lease, (2) New EPA regulations require $200M environmental remediation, (3) Compliance costs make operation unprofitable, (4) MinerCorp claims regulatory taking requiring compensation, (5) EPA cites police power for environmental protection without compensation. ISSUE: Whether environmental regulations requiring expensive compliance constitute a taking under Fifth Amendment. RULE: Regulatory taking occurs when regulation denies all economically viable use or fails to substantially advance legitimate government interest; environmental protection is valid police power; economic impact alone insufficient for taking claim. APPLICATION: $200M cost against profitability suggests severe economic impact; environmental protection constitutes legitimate government interest; must examine whether any economically viable use remains; police power regulations generally don't require compensation unless destroying all economic value. CONCLUSION: Likely not a taking unless regulations eliminate all economic value; environmental compliance costs alone insufficient for compensation. TAKINGS ANALYSIS: Government regulation vs. physical occupation; police power presumptively valid for environmental protection; total economic destruction required for regulatory taking.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "environmental_law"
        },
        "extra_info": {
        "split": "train",
        "index": 186,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Lucas v. South Carolina Coastal Council, 505 U.S. 1003 (1992)", "Penn Coal Co. v. Mahon, 260 U.S. 393 (1922)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this entertainment law contract dispute: Recording artist Maya signed exclusive 5-album deal with RecordCorp for $2 million advance. After releasing 2 successful albums, Maya wants to terminate the contract claiming artistic differences and inadequate promotion. The contract includes specific marketing budget requirements that RecordCorp met, but Maya argues the promotional strategy was inappropriate for her artistic vision. RecordCorp seeks injunction preventing Maya from recording elsewhere."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Maya: exclusive 5-album deal, $2M advance from RecordCorp, (2) 2 successful albums completed, 3 remaining, (3) Maya seeks termination for artistic differences and inadequate promotion, (4) Contract marketing budgets were met by RecordCorp, (5) Maya claims promotional strategy inappropriate for artistic vision, (6) RecordCorp seeks injunction against recording elsewhere. ISSUE: Whether artistic differences and promotional disagreements justify contract termination and whether injunctive relief is appropriate. RULE: Contract performance requires substantial compliance with material terms; artistic differences alone insufficient for breach unless contract gives artist control over promotion; personal services contracts may not be specifically enforced against artist but may prevent performance for competitors. APPLICATION: RecordCorp met contractual marketing budget obligations; promotional strategy disputes don't constitute material breach absent contractual creative control provisions; Maya received $2M advance creating obligation to complete remaining albums; negative injunction may be appropriate to prevent competing performance. CONCLUSION: Contract termination likely invalid; RecordCorp may obtain negative injunction preventing Maya from recording elsewhere but cannot force performance. ARTISTIC CONTROL: Unless contract grants artist promotional control, label decisions on marketing strategy within budget parameters don't constitute breach.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "entertainment_law"
        },
        "extra_info": {
        "split": "train",
        "index": 187,
        "difficulty": "advanced",
        "court_level": "state_trial",
        "case_citations": ["Lumley v. Wagner, 42 Eng. Rep. 687 (1852)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this energy law regulatory challenge: PowerCorp operates nuclear plant under NRC license expiring in 2025. PowerCorp applied for 20-year license renewal, but environmental groups challenge the application claiming inadequate analysis of climate change impacts and aging infrastructure risks. NRC approved renewal based on standard safety review. Environmental groups argue new climate data requires enhanced environmental impact analysis under NEPA."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) PowerCorp nuclear plant license expires 2025, (2) Applied for 20-year renewal, (3) Environmental groups challenge for inadequate climate change and aging infrastructure analysis, (4) NRC approved based on standard safety review, (5) Groups claim new climate data requires enhanced NEPA analysis. ISSUE: Whether NRC's environmental analysis under NEPA adequately considered climate change impacts and aging infrastructure in license renewal. RULE: NEPA requires environmental impact analysis for major federal actions; analysis must consider reasonably foreseeable environmental effects; agencies must consider new and significant information; arbitrary and capricious standard applies to agency NEPA compliance. APPLICATION: 20-year renewal constitutes major federal action triggering NEPA; climate change impacts over renewal period reasonably foreseeable; aging infrastructure presents safety concerns requiring analysis; new climate data may constitute significant information requiring enhanced review. CONCLUSION: Challenge may succeed if climate change analysis inadequate or new data not properly considered; courts defer to agency expertise but require adequate NEPA compliance. NEPA SCOPE: Environmental analysis must be comprehensive and consider all reasonably foreseeable impacts; arbitrary and capricious review ensures adequate consideration.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "energy_law"
        },
        "extra_info": {
        "split": "train",
        "index": 188,
        "difficulty": "advanced",
        "court_level": "federal_circuit",
        "case_citations": ["National Environmental Policy Act 42 U.S.C. § 4321", "Marsh v. Oregon Natural Resources Council, 490 U.S. 360 (1989)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this international trade law dispute: U.S. imposed 25% tariffs on steel imports from Country X claiming national security justification under Section 232. Country X challenges the tariffs at WTO, arguing they constitute protectionist trade barriers rather than legitimate security measures. The U.S. steel industry supports the tariffs claiming foreign competition threatens domestic production capacity needed for defense manufacturing."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) U.S. imposed 25% steel tariffs on Country X imports, (2) Justified under Section 232 national security authority, (3) Country X challenges at WTO as protectionist barriers, (4) U.S. steel industry supports tariffs citing defense manufacturing needs, (5) Dispute over legitimate security vs. economic protection. ISSUE: Whether steel tariffs constitute legitimate national security measure or protectionist trade barrier under WTO rules. RULE: WTO allows national security exceptions for measures protecting essential security interests; exceptions narrowly construed to prevent abuse; economic protection alone insufficient for security exception; genuine nexus between measure and security required. APPLICATION: Steel production has defense implications but tariffs must address actual security threat, not economic competition; broad tariff coverage suggests economic rather than security motivation; domestic industry support indicates protectionist rather than security purpose; security exception requires genuine threat analysis, not general industry protection. CONCLUSION: Tariffs likely violate WTO rules unless genuine security nexus demonstrated; economic protection insufficient for security exception. SECURITY EXCEPTION: Must address actual threat to essential security interests, not general economic competitiveness; WTO panels review security claims for good faith and genuine nexus.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "international",
        "legal_domain": "trade_law"
        },
        "extra_info": {
        "split": "train",
        "index": 189,
        "difficulty": "advanced",
        "court_level": "international_tribunal",
        "case_citations": ["GATT Article XXI", "WTO Panel Reports on Security Exceptions"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this telecommunications law case: MobileCorp and rival WirelessInc agreed to share cell tower infrastructure to reduce costs. The agreement included non-compete clauses preventing either company from building competing towers within 5 miles of shared facilities. FCC investigates the arrangement claiming it reduces competition and violates telecommunications regulations requiring infrastructure access. Companies argue the sharing promotes efficiency and reduces environmental impact."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) MobileCorp and WirelessInc share cell tower infrastructure, (2) Agreement includes 5-mile non-compete clauses around shared towers, (3) FCC investigates for competition reduction and regulation violations, (4) Companies cite cost efficiency and environmental benefits, (5) Dispute over competition vs. efficiency in infrastructure sharing. ISSUE: Whether infrastructure sharing with non-compete provisions violates telecommunications competition requirements. RULE: Telecommunications regulations promote competition and infrastructure access; agreements restraining competition require careful antitrust analysis; efficiencies may justify some competitive restraints if benefits outweigh harms; FCC has authority to regulate anticompetitive practices. APPLICATION: Infrastructure sharing creates efficiencies and environmental benefits; 5-mile non-compete clauses may reduce site competition and consumer choice; must balance efficiency gains against competitive restrictions; FCC regulations favor open access and competition. CONCLUSION: Agreement may violate competition rules unless efficiency benefits clearly outweigh competitive harms; non-compete clauses particularly problematic. TELECOMMUNICATIONS POLICY: Infrastructure sharing encouraged but not at expense of meaningful competition; efficiency justifications must be substantial and verifiable. BALANCING TEST: Competitive benefits vs. efficiency gains requires case-by-case analysis of market impact.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "telecommunications_law"
        },
        "extra_info": {
        "split": "train",
        "index": 190,
        "difficulty": "advanced",
        "court_level": "federal_commission",
        "case_citations": ["Communications Act of 1934", "Telecommunications Act of 1996"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this maritime law salvage case: Cargo ship MV Prosperity ran aground in storm 50 miles offshore carrying $100 million in electronics. Commercial salvage vessel Ocean Rescue responded to distress call and successfully refloated the ship after 3 days, bringing it safely to port. Ship's owners refuse to pay the $10 million salvage claim, arguing Ocean Rescue was already contracted for routine towing and had no right to salvage award."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) MV Prosperity grounded in storm 50 miles offshore, (2) $100M electronics cargo aboard, (3) Ocean Rescue responded to distress call, (4) Successfully refloated ship after 3 days, (5) Ship brought safely to port, (6) Owners refuse $10M salvage claim citing existing towing contract. ISSUE: Whether Ocean Rescue entitled to salvage award despite pre-existing towing contract. RULE: Maritime salvage requires: (1) marine peril, (2) voluntary service, (3) success in saving property; salvage award based on danger, value saved, effort expended; existing contracts may limit salvage rights if services within contract scope. APPLICATION: Grounding in storm 50 miles offshore constitutes marine peril; successful refloating and safe port delivery shows success; $100M cargo value supports substantial award; existing towing contract may limit salvage rights if emergency services within contracted scope. CONCLUSION: Salvage award likely unless towing contract specifically covered emergency grounding situations; storm conditions and distance suggest services beyond routine towing. SALVAGE ELEMENTS: All three elements clearly met; contractual limitation requires analysis of service scope. AWARD CALCULATION: Based on danger, skill required, value saved, and time expended - $10M claim reasonable for $100M cargo rescue.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "maritime_law"
        },
        "extra_info": {
        "split": "train",
        "index": 191,
        "difficulty": "intermediate",
        "court_level": "federal_district",
        "case_citations": ["The Blackwall, 77 U.S. 1 (1869)", "General Maritime Law"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this gaming law regulation challenge: State legislature banned online poker while allowing other forms of internet gambling like fantasy sports and casino games. PokerSite challenges the law claiming arbitrary discrimination that violates equal protection. State argues poker involves more skill than chance compared to pure gambling, making regulation appropriate. PokerSite contends poker requires as much skill as permitted fantasy sports."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) State banned online poker while allowing fantasy sports and casino games, (2) PokerSite challenges as arbitrary discrimination, (3) Equal protection violation claimed, (4) State argues poker involves more skill than chance compared to permitted gambling, (5) PokerSite claims poker skill level comparable to permitted fantasy sports. ISSUE: Whether selective prohibition of online poker while allowing other internet gambling violates equal protection. RULE: Equal protection requires rational basis for classification; arbitrary distinctions without legitimate purpose violate constitutional rights; gambling regulation must have reasonable relationship to state interests. APPLICATION: Poker and fantasy sports both involve skill and chance elements; selective prohibition appears arbitrary if skill levels comparable; state interest in preventing gambling undermined by allowing casino games; rational basis test requires logical distinction between permitted and prohibited activities. CONCLUSION: Classification likely fails rational basis test due to arbitrary distinction between similar skill-based games; selective enforcement undermines stated regulatory purpose. EQUAL PROTECTION ANALYSIS: Similar activities must receive similar treatment absent rational justification; skill vs. chance distinction must be logical and consistent. GAMBLING REGULATION: States may regulate gambling but classifications must be reasonable and non-arbitrary.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "gaming_law"
        },
        "extra_info": {
        "split": "train",
        "index": 192,
        "difficulty": "intermediate",
        "court_level": "federal_district",
        "case_citations": ["City of Cleburne v. Cleburne Living Center, 473 U.S. 432 (1985)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this food safety law case: RestaurantCorp operates 200 locations nationwide serving pre-packaged salads from central facility. FDA investigation after foodborne illness outbreak traced contamination to the central processing facility. 500 customers became ill across 15 states. RestaurantCorp claims individual franchise locations are responsible for food safety, not the corporate entity. FDA seeks to hold corporation liable under food safety modernization act."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) RestaurantCorp operates 200 locations with centralized salad facility, (2) FDA traced foodborne illness outbreak to central processing, (3) 500 customers ill across 15 states, (4) RestaurantCorp claims franchise locations responsible for safety, (5) FDA seeks corporate liability under food safety modernization act. ISSUE: Whether corporate entity liable for food safety violations at centralized processing facility serving franchise locations. RULE: Food Safety Modernization Act imposes strict liability on food processors; corporate responsibility extends to controlled facilities; interstate commerce triggers federal jurisdiction; franchise structure doesn't eliminate corporate liability for controlled operations. APPLICATION: Central processing facility under corporate control establishes direct liability; 15-state distribution shows interstate commerce jurisdiction; 500 illnesses demonstrate substantial public health impact; franchise defense fails when corporation controls contamination source. CONCLUSION: Corporation likely liable under FSMA for central facility contamination; franchise structure doesn't insulate from liability for controlled processing operations. CORPORATE LIABILITY: Control over contamination source creates direct responsibility regardless of distribution method. FSMA SCOPE: Strict liability for food processors with interstate distribution; corporate structure irrelevant to safety obligations.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "food_safety_law"
        },
        "extra_info": {
        "split": "train",
        "index": 193,
        "difficulty": "intermediate",
        "court_level": "federal_district",
        "case_citations": ["Food Safety Modernization Act 21 U.S.C. § 350g"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC to analyze this animal welfare law enforcement action: PetStore chains operates 50 locations selling puppies sourced from commercial breeding facilities. Animal welfare group investigation revealed overcrowded conditions, inadequate veterinary care, and high puppy mortality rates at supplier facilities. State attorney general sues under animal cruelty statute claiming PetStore knew of conditions but continued purchasing. PetStore argues they relied on facility licenses and had no direct knowledge of violations."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) PetStore operates 50 locations selling puppies from commercial breeders, (2) Investigation revealed overcrowded conditions and inadequate vet care, (3) High puppy mortality rates at supplier facilities, (4) State sues under animal cruelty statute for continuing purchases despite knowledge, (5) PetStore claims reliance on facility licenses without direct knowledge. ISSUE: Whether retailer liable under animal cruelty statute for continuing business with suppliers engaged in animal welfare violations. RULE: Animal cruelty statutes may impose liability for aiding or enabling cruelty; willful blindness to known conditions may constitute knowledge; commercial relationships create duties of reasonable investigation for ongoing violations. APPLICATION: High mortality rates and investigation findings provide notice of conditions; continued purchasing after notice may constitute enabling cruelty; facility licenses don't eliminate duty to investigate known problems; 50-location chain suggests commercial sophistication requiring due diligence. CONCLUSION: Liability likely if PetStore continued purchasing after reasonable notice of welfare violations; license reliance insufficient defense against actual knowledge. ENABLING LIABILITY: Continued commercial relationship after notice of violations may constitute aiding animal cruelty. COMMERCIAL DUTY: Large retailers may have enhanced obligations to monitor supplier practices affecting animal welfare.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "general",
        "legal_domain": "animal_welfare_law"
        },
        "extra_info": {
        "split": "train",
        "index": 194,
        "difficulty": "intermediate",
        "court_level": "state_trial",
        "case_citations": ["State Animal Cruelty Statutes", "Animal Welfare Act 7 U.S.C. § 2131"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC to this disability rights accommodation case: Employee Jennifer, who has multiple sclerosis, requested to work from home 3 days per week as reasonable accommodation. Employer TechFirm denied the request claiming in-person collaboration essential for her software development team. Jennifer's productivity metrics show above-average performance, and similar positions at other companies routinely work remotely. TechFirm offers alternative accommodations like flexible hours and ergonomic equipment."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Jennifer has multiple sclerosis, (2) Requested 3-day remote work accommodation, (3) TechFirm denied citing essential in-person collaboration, (4) Jennifer's performance metrics above average, (5) Similar positions elsewhere work remotely, (6) Alternative accommodations offered (flexible hours, ergonomic equipment). ISSUE: Whether remote work constitutes reasonable accommodation and whether denial violates ADA obligations. RULE: ADA requires reasonable accommodation unless undue hardship or fundamental job function alteration; employer must engage in interactive process; accommodation must enable essential job performance; alternatives may satisfy obligation if effective. APPLICATION: Remote work increasingly recognized as reasonable accommodation for knowledge workers; above-average performance suggests essential functions can be performed remotely; industry practice supports feasibility; employer must prove in-person collaboration truly essential vs. preferred; alternative accommodations may be inadequate if remote work needed for disability management. CONCLUSION: Remote work likely required as reasonable accommodation unless employer proves undue hardship or essential function conflict; offered alternatives insufficient if they don't address disability-related limitations. ADA STANDARD: Accommodation must be effective for individual's specific disability needs; employer preference insufficient to deny reasonable requests.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "disability_law"
        },
        "extra_info": {
        "split": "train",
        "index": 195,
        "difficulty": "intermediate",
        "court_level": "federal_district",
        "case_citations": ["Americans with Disabilities Act 42 U.S.C. § 12111"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze using FIRAC this pharmaceutical patent case: DrugCorp holds patent on cancer medication expected to expire in 2027. Generic manufacturer BioTech filed abbreviated new drug application (ANDA) claiming DrugCorp's patent is invalid due to prior art showing similar compounds existed before filing date. DrugCorp sued for patent infringement and seeks preliminary injunction preventing generic entry. BioTech argues cancer patients need affordable access and prior art invalidates exclusivity claims."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) DrugCorp holds cancer drug patent expiring 2027, (2) BioTech filed ANDA for generic version, (3) BioTech claims patent invalid due to prior art showing similar compounds, (4) DrugCorp sues for infringement seeking preliminary injunction, (5) BioTech argues patient access need and prior art invalidity. ISSUE: Whether preliminary injunction appropriate when patent validity challenged based on prior art claims. RULE: Preliminary injunction requires: (1) likelihood of success on merits, (2) irreparable harm, (3) balance of hardships, (4) public interest; patent validity presumed but prior art may overcome presumption; Hatch-Waxman Act provides framework for generic challenges. APPLICATION: Patent validity challenged by credible prior art evidence; DrugCorp faces irreparable harm from generic competition; BioTech and patients benefit from affordable access; public interest includes both innovation incentives and healthcare access; strength of prior art evidence affects likelihood of success. CONCLUSION: Injunction depends on strength of prior art challenge; weak invalidity claims favor injunction, strong prior art evidence may defeat preliminary relief. PHARMACEUTICAL BALANCE: Innovation protection vs. patient access requires careful analysis of patent strength and public health impact.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "pharmaceutical_law"
        },
        "extra_info": {
        "split": "train",
        "index": 196,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Hatch-Waxman Act 21 U.S.C. § 355", "eBay Inc. v. MercExchange, 547 U.S. 388 (2006)"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this agricultural law water dispute: Farmer Coalition represents 100 agricultural operations claiming state water allocation system unfairly prioritizes urban development over farming needs. State reduced agricultural water allocations by 40% while maintaining full allocations for residential and commercial development. Farmers argue this violates constitutional equal protection and takes their property without just compensation. State claims urban needs are more critical during drought conditions."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) Farmer Coalition represents 100 agricultural operations, (2) State water allocation reduced agriculture by 40%, (3) Urban development maintained full allocations, (4) Farmers claim equal protection violation and taking without compensation, (5) State argues urban needs more critical during drought. ISSUE: Whether differential water allocation between agriculture and urban development violates constitutional protections. RULE: Equal protection requires rational basis for different treatment; regulatory taking occurs when government action destroys economic value; water rights may constitute property interests; emergency powers allow temporary restrictions with appropriate justification. APPLICATION: 40% agricultural reduction vs. full urban allocation suggests disparate treatment; rational basis requires justification for differential treatment; water rights as property interests may trigger taking analysis; drought emergency may justify some restrictions but must be reasonably related to crisis. CONCLUSION: Classification likely violates equal protection absent compelling justification for treating agriculture differently; potential taking if water rights constitute protected property. WATER ALLOCATION: Equal treatment principle applies unless emergency justifies differential restrictions; urban vs. agricultural distinction requires rational policy basis.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "agricultural_law"
        },
        "extra_info": {
        "split": "train",
        "index": 197,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Takings Clause U.S. Const. Amend. V", "Equal Protection Clause U.S. Const. Amend. XIV"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Apply FIRAC reasoning to this tribal law jurisdiction dispute: Native American John committed assault on tribal land against non-tribal member victim Sarah. Federal prosecutors want to charge John under federal law, while tribal court claims exclusive jurisdiction under tribal sovereignty principles. Victim Sarah prefers federal prosecution believing tribal penalties inadequate. Defense argues tribal court provides culturally appropriate justice and federal intervention violates sovereignty."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) John (Native American) assaulted Sarah (non-tribal member) on tribal land, (2) Federal prosecutors seek federal charges, (3) Tribal court claims exclusive jurisdiction under sovereignty, (4) Victim prefers federal prosecution citing inadequate tribal penalties, (5) Defense argues tribal court culturally appropriate and federal intervention violates sovereignty. ISSUE: Whether tribal or federal court has jurisdiction over tribal member's assault against non-tribal victim on tribal land. RULE: Tribal sovereignty grants criminal jurisdiction over tribal members on tribal land; federal government has jurisdiction over major crimes under Major Crimes Act; jurisdiction may be concurrent; victim preference doesn't determine jurisdiction. APPLICATION: Tribal member status and tribal land location support tribal jurisdiction; assault may qualify as major crime under federal jurisdiction; tribal sovereignty principles favor tribal court authority; jurisdictional analysis focuses on sovereignty and statutory authority, not victim preference or penalty adequacy. CONCLUSION: Likely concurrent jurisdiction with tribal court having primary authority; federal prosecution possible if assault constitutes major crime under federal statute. SOVEREIGNTY BALANCE: Tribal self-governance vs. federal criminal law enforcement requires careful jurisdictional analysis based on legal authority, not policy preferences.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "tribal_law"
        },
        "extra_info": {
        "split": "train",
        "index": 198,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Major Crimes Act 18 U.S.C. § 1153", "Indian Self-Determination Act"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Analyze with FIRAC this space law commercial dispute: SpaceCorp launched private satellite constellation for internet service but orbital debris from their failed rocket damaged competitor SatelliteInc's existing satellites. International space law requires liability for space debris, but SpaceCorp argues the damage was unforeseeable accident beyond their control. SatelliteInc seeks $500 million compensation for lost satellites and service disruption under space liability treaties."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) SpaceCorp launched satellite constellation for internet service, (2) Failed rocket created orbital debris, (3) Debris damaged SatelliteInc's existing satellites, (4) International space law requires liability for space debris, (5) SpaceCorp claims unforeseeable accident defense, (6) SatelliteInc seeks $500M compensation under space liability treaties. ISSUE: Whether space debris liability applies despite claimed unforeseeable accident circumstances. RULE: Outer Space Treaty and Liability Convention impose strict liability for space activities; launching state responsible for damage caused by space objects; fault or foreseeability generally irrelevant for liability; compensation required for actual damages. APPLICATION: SpaceCorp's failed rocket constitutes space object under treaty definitions; debris damage to existing satellites clearly within liability scope; strict liability standard eliminates foreseeability defense; $500M claim requires assessment of actual satellite value and service disruption damages. CONCLUSION: SpaceCorp likely liable under space treaties regardless of accident characterization; compensation required for proven damages. SPACE LIABILITY: Strict liability regime for space activities reflects high-risk nature of space operations; fault-based defenses generally unavailable. DAMAGE CALCULATION: Compensation includes satellite replacement costs and quantifiable service disruption losses.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "international",
        "legal_domain": "space_law"
        },
        "extra_info": {
        "split": "train",
        "index": 199,
        "difficulty": "advanced",
        "court_level": "international_tribunal",
        "case_citations": ["Outer Space Treaty 1967", "Convention on International Liability for Damage Caused by Space Objects 1972"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    },
    {
        "data_source": "legal_reasoning",
        "prompt": [
        {
            "role": "user",
            "content": "Use FIRAC for this artificial intelligence regulation case: TechGiant deployed AI hiring algorithm that screened job applications for major corporation. Algorithm systematically rejected qualified female candidates due to training data bias from historical hiring patterns. Rejected applicants sue under employment discrimination laws claiming disparate impact. TechGiant argues algorithm was neutral and any bias resulted from legitimate historical data, not intentional discrimination."
        }
        ],
        "reward_model": {
        "style": "multi_task_legal",
        "ground_truth": "FACTS: (1) TechGiant's AI hiring algorithm screened applications for corporation, (2) Algorithm systematically rejected qualified female candidates, (3) Bias resulted from historical hiring pattern training data, (4) Rejected applicants sue for employment discrimination under disparate impact theory, (5) TechGiant claims algorithm neutrality and legitimate historical data use. ISSUE: Whether AI algorithm disparate impact on protected class violates employment discrimination laws despite neutral design. RULE: Title VII prohibits employment practices with disparate impact on protected classes; employer must prove business necessity and job-relatedness; neutral policies may violate law if discriminatory effect; intent unnecessary for disparate impact claims. APPLICATION: Systematic rejection of qualified female candidates establishes disparate impact; historical bias data perpetuates past discrimination; algorithm's neutral appearance doesn't excuse discriminatory results; employer must prove hiring algorithm job-related and business necessity; available alternative methods without discriminatory impact. CONCLUSION: Strong disparate impact case against AI algorithm; historical data bias defense insufficient under employment discrimination law. AI DISCRIMINATION: Algorithmic bias extends traditional discrimination concepts to automated decision-making; neutral technology may perpetuate historical discrimination patterns requiring legal remedy.",
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "ai_regulation_law"
        },
        "extra_info": {
        "split": "train",
        "index": 200,
        "difficulty": "advanced",
        "court_level": "federal_district",
        "case_citations": ["Griggs v. Duke Power Co., 401 U.S. 424 (1971)", "Title VII 42 U.S.C. § 2000e"],
        "tools_available": ["courtlistener", "brave_search"]
        }
    }
]



test_data = [
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Analyze the precedential value of Miranda v. Arizona (1966) for this scenario: Police arrested suspect without Miranda warnings, but suspect spontaneously confessed before any questioning began. How do courts typically apply Miranda precedent to spontaneous statements, and what distinguishing factors have emerged in subsequent case law?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Miranda v. Arizona requires warnings before custodial interrogation, but courts have consistently distinguished spontaneous statements. Key precedent analysis shows: (1) Rhode Island v. Innis limits Miranda to interrogation, not spontaneous utterances; (2) Illinois v. Perkins shows undercover context distinctions; (3) Circuits split on definitional boundaries of 'spontaneous.' Analogical reasoning demonstrates Miranda's interrogation requirement doesn't extend to voluntary statements absent police questioning.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 201,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Miranda v. Arizona, 384 U.S. 436 (1966)", "Rhode Island v. Innis, 446 U.S. 291 (1980)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft a legal opinion for our client, a small business owner, explaining their rights and potential remedies after a competitor opened a nearly identical business using confidential information obtained from a former employee who violated their non-compete agreement. Focus on trade secret protection and unfair competition claims."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Legal opinion should address: (1) Trade secret misappropriation elements under state UTSA; (2) Non-compete enforcement analysis including reasonableness factors; (3) Unfair competition claims for improper business advantage; (4) Available remedies including injunctive relief and damages; (5) Practical litigation strategy and evidence preservation recommendations. Client-focused language explaining legal options and recommended actions.",
      "task_type": "opinion_generation",
      "jurisdiction": "general",
      "legal_domain": "business_law"
    },
    "extra_info": {
      "split": "test",
      "index": 202,
      "difficulty": "intermediate",
      "court_level": "state_trial",
      "case_citations": ["Uniform Trade Secrets Act"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I'm a college student who was injured when a defective chair collapsed during class. The chair was manufactured by FurnitureCorp and purchased by the university. What types of legal claims might I have, and what should I know about product liability law? I'm not sure if I should sue the manufacturer, the university, or both."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "You may have several potential claims: (1) Product liability against FurnitureCorp for defective design/manufacturing; (2) Premises liability against university for unsafe conditions; (3) Negligence claims against both parties. Product liability allows recovery without proving manufacturer negligence if chair was unreasonably dangerous. University liability depends on notice of defect and duty to maintain safe premises. Consider factors like injury severity, medical expenses, and available insurance. Recommend consulting personal injury attorney for case evaluation.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "tort_law"
    },
    "extra_info": {
      "split": "test",
      "index": 203,
      "difficulty": "beginner",
      "court_level": "educational",
      "case_citations": ["Restatement (Third) of Torts: Products Liability"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this employment termination: Software engineer Lisa was fired after reporting gender pay disparities to HR. Company claims termination was due to missing project deadlines, but Lisa had received positive performance reviews and the deadline issues began only after her complaint. Lisa's male colleagues with similar or worse performance were not terminated. Analyze potential retaliation and discrimination claims."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Lisa reported gender pay disparities to HR, (2) Terminated for allegedly missing deadlines, (3) Positive performance reviews previously, (4) Deadline issues began post-complaint, (5) Male colleagues with similar/worse performance retained. ISSUE: Whether termination constitutes unlawful retaliation and gender discrimination. RULE: Title VII prohibits retaliation for protected complaints; employers cannot terminate for opposing discriminatory practices. APPLICATION: Pay disparity complaint = protected activity; temporal proximity between complaint and termination suggests causal connection; pretextual performance reasons evidenced by timing and differential treatment. CONCLUSION: Strong retaliation case with supportive circumstantial evidence; potential discrimination claim based on disparate treatment.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "employment_law"
    },
    "extra_info": {
      "split": "test",
      "index": 204,
      "difficulty": "intermediate",
      "court_level": "federal_district",
      "case_citations": ["Burlington Northern v. White, 548 U.S. 53 (2006)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Compare how the Supreme Court's decision in Kelo v. City of New London (2005) has been interpreted by different circuit courts regarding economic development takings. What analogical reasoning patterns emerge when lower courts distinguish or follow Kelo in subsequent eminent domain cases?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Circuit courts show varied interpretations of Kelo's 'public use' expansion: (1) 2nd and 9th Circuits broadly apply economic development rationale; (2) 4th and 5th Circuits require stronger public benefit showing; (3) State courts increasingly reject Kelo through constitutional interpretation or legislation. Distinguishing factors include: economic necessity evidence, private party beneficiary prominence, and genuine public planning process. Analogical reasoning focuses on public purpose legitimacy versus private party enrichment.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 205,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Kelo v. City of New London, 545 U.S. 469 (2005)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "My landlord is trying to evict me for having a pet, but my lease doesn't mention pets at all. I have a registered emotional support animal that my therapist prescribed for anxiety. The landlord says emotional support animals aren't real service animals. What are my rights regarding housing discrimination and emotional support animals?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "You have protections under Fair Housing Act for emotional support animals, which are different from service animals. Key points: (1) ESAs are reasonable accommodations for disabilities, not service animals under ADA; (2) Landlords must allow ESAs even with 'no pets' policies unless undue hardship; (3) You need documentation from healthcare provider; (4) Landlords cannot charge pet fees for ESAs. If lease silent on pets, stronger position for accommodation. Request written accommodation with therapist documentation. Landlord education about FHA requirements may resolve issue without legal action.",
      "task_type": "general_chat",
      "jurisdiction": "federal",
      "legal_domain": "housing_law"
    },
    "extra_info": {
      "split": "test",
      "index": 206,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Fair Housing Act 42 U.S.C. § 3604"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare a legal memorandum for corporate client explaining their potential liability exposure after a data breach compromised 100,000 customer records containing personal information and payment data. Address regulatory compliance, private litigation risks, and recommended response strategy under applicable privacy laws."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Memorandum should address: (1) State data breach notification requirements and timing; (2) Potential FTC enforcement under unfair practices authority; (3) State privacy law violations and damages exposure; (4) Class action litigation risks and typical settlement ranges; (5) Credit monitoring obligations and mitigation costs; (6) Insurance coverage analysis; (7) Immediate response protocol including forensic investigation, notification strategy, and regulatory cooperation. Emphasize compliance timing and documentation requirements.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "privacy_law"
    },
    "extra_info": {
      "split": "test",
      "index": 207,
      "difficulty": "advanced",
      "court_level": "regulatory",
      "case_citations": ["FTC Act Section 5", "State Data Breach Notification Laws"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this contract dispute: Homeowner hired contractor to renovate kitchen for $75,000 with completion by December 1st. Contract included $500/day penalty for late completion. Contractor finished on December 15th due to supply chain delays beyond their control, but homeowner had already hired another contractor to finish the work. Original contractor demands payment minus penalty; homeowner refuses any payment claiming material breach."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) $75k kitchen renovation contract with Dec 1 completion, (2) $500/day late penalty clause, (3) Contractor finished Dec 15 due to supply chain delays, (4) Homeowner hired second contractor before completion, (5) Original contractor seeks payment minus penalty, (6) Homeowner claims material breach. ISSUE: Whether 15-day delay constitutes material breach excusing payment obligations. RULE: Material breach analysis considers importance of timely performance, extent of delay, and substantial performance doctrine. APPLICATION: 15-day delay significant but not necessarily material if work substantially performed; supply chain delays may excuse some delays; homeowner's hiring second contractor suggests anticipatory repudiation. CONCLUSION: Likely not material breach requiring forfeiture; contractor entitled to payment minus liquidated damages ($7,500) if work substantially performed.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "general",
      "legal_domain": "contract_law"
    },
    "extra_info": {
      "split": "test",
      "index": 208,
      "difficulty": "intermediate",
      "court_level": "state_trial",
      "case_citations": ["Restatement (Second) of Contracts § 241"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Trace the evolution of the business judgment rule through key Delaware cases, analyzing how courts have applied precedent in hostile takeover scenarios. Compare the reasoning in Unocal Corp. v. Mesa Petroleum Co. (1985) with subsequent cases and identify trends in judicial application."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Unocal established enhanced scrutiny for defensive measures: (1) reasonable belief of threat, (2) proportionate response. Evolution shows: Moran v. Household (1985) applied Unocal to poison pills; Revlon (1986) added auction duties; Paramount v. Time (1989) clarified Unocal/Revlon boundaries; Unitrin (1995) refined proportionality test. Pattern shows Delaware courts balancing director authority with shareholder protection, with enhanced scrutiny becoming standard for conflicts of interest while preserving business judgment protection for ordinary decisions.",
      "task_type": "precedent_analysis",
      "jurisdiction": "Delaware",
      "legal_domain": "corporate_law"
    },
    "extra_info": {
      "split": "test",
      "index": 209,
      "difficulty": "advanced",
      "court_level": "state_supreme",
      "case_citations": ["Unocal Corp. v. Mesa Petroleum Co., 493 A.2d 946 (Del. 1985)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I received a demand letter claiming I violated someone's trademark by using a similar business name. My business is 'Quick Clean Auto Detailing' and they claim it infringes their 'QuickClean Car Wash' trademark. Is this a valid trademark infringement claim? What factors determine trademark infringement, and what should I do?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Trademark infringement requires likelihood of consumer confusion analysis using factors like: (1) similarity of marks, (2) similarity of goods/services, (3) channels of trade, (4) consumer sophistication, (5) strength of prior mark. 'Quick Clean' vs 'QuickClean' are similar, and auto detailing vs car wash are related services, suggesting potential confusion. However, geographic factors, actual use dates, and federal registration status matter. Respond promptly to demand letter, consider trademark search, document your independent development, and evaluate settlement vs. defense options. Consult IP attorney for formal analysis.",
      "task_type": "general_chat",
      "jurisdiction": "federal",
      "legal_domain": "intellectual_property"
    },
    "extra_info": {
      "split": "test",
      "index": 210,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Lanham Act 15 U.S.C. § 1114"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft a persuasive brief section arguing that our client's First Amendment rights were violated when the city denied their permit application for a political protest based on the controversial nature of their message. Address content-based restrictions and apply strict scrutiny analysis."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Brief should argue: (1) Permit denial based on message controversy constitutes content-based restriction triggering strict scrutiny; (2) Public forums require viewpoint neutrality; controversial content cannot justify permit denial; (3) No compelling government interest in suppressing political speech; (4) Denial was not narrowly tailored - alternative measures available for safety concerns; (5) Prior restraint doctrine requires procedural safeguards absent here. Cite Reed v. Town of Gilbert for content neutrality requirements and emphasize fundamental nature of political speech in democratic society.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 211,
      "difficulty": "advanced",
      "court_level": "federal_district",
      "case_citations": ["Reed v. Town of Gilbert, 576 U.S. 155 (2015)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this criminal sentencing issue: Defendant Jones pleaded guilty to federal drug trafficking with 100 grams of heroin. Under federal guidelines, this triggers a 10-year mandatory minimum. Defense argues for downward departure citing Jones's role as minor participant and extraordinary family circumstances (sole caregiver for disabled child). Government opposes any departure from mandatory minimum."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Jones guilty of trafficking 100g heroin, (2) Federal guidelines mandate 10-year minimum, (3) Defense seeks downward departure for minor participant role, (4) Extraordinary family circumstances - disabled child caregiving, (5) Government opposes departure from mandatory sentence. ISSUE: Whether minor participant role and family circumstances justify departure from mandatory minimum. RULE: Mandatory minimums generally prohibit judicial departure; minor participant adjustment limited; extraordinary family circumstances rarely sufficient for departure below statutory minimum. APPLICATION: 100g heroin triggers absolute minimum regardless of role; family circumstances, while sympathetic, don't constitute legal basis for departure below mandatory minimum; minor participant argument unavailable for mandatory minimum sentences. CONCLUSION: Court likely bound by 10-year mandatory minimum despite mitigating factors; limited discretion for departure below statutory floor.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "criminal_law"
    },
    "extra_info": {
      "split": "test",
      "index": 212,
      "difficulty": "advanced",
      "court_level": "federal_district",
      "case_citations": ["21 U.S.C. § 841", "U.S. Sentencing Guidelines § 3B1.2"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Analyze how different circuits have interpreted the Supreme Court's decision in Carpenter v. United States (2018) regarding digital privacy rights. What precedential patterns emerge in cases involving location tracking technology, and how do courts distinguish between different types of digital surveillance?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Carpenter established Fourth Amendment protection for historical cell-site location information, but circuits diverge on scope: (1) 2nd and 9th Circuits broadly apply to real-time tracking and shorter duration requests; (2) 5th and 11th Circuits limit to historical data exceeding week-long periods; (3) Circuits split on third-party doctrine application to other digital records. Distinguishing factors include: duration of surveillance, real-time vs. historical data, precision of location information, and voluntary disclosure analysis. Pattern shows expansion of privacy rights for comprehensive digital surveillance while maintaining traditional third-party doctrine for limited data requests.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 213,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Carpenter v. United States, 138 S. Ct. 2206 (2018)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "My elderly parent signed a contract to buy a $50,000 timeshare after a high-pressure sales presentation. They have dementia that's been getting worse, but they weren't legally declared incompetent. The contract has a 3-day cancellation period that already passed. Can we still cancel this contract? What legal options exist for elderly financial abuse situations?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Several potential options exist: (1) Lack of mental capacity defense - contracts voidable if person lacked understanding due to dementia; (2) Undue influence claim if high-pressure tactics exploited vulnerability; (3) State consumer protection laws often provide extended cancellation periods for seniors; (4) Elder abuse statutes may apply with enhanced penalties and remedies; (5) Unconscionability doctrine for grossly unfair terms. Document medical condition, gather evidence of sales tactics, and act quickly. Many states have specific protections for elderly consumers beyond standard contract law. Consult elder law attorney immediately.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "elder_law"
    },
    "extra_info": {
      "split": "test",
      "index": 214,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Restatement (Second) of Contracts § 15"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare a legal opinion letter to municipal client addressing their liability exposure for police officer's use of excessive force during arrest. Officer deviated from department policy, but city argues qualified immunity protects both officer and municipality. Analyze Section 1983 liability, municipal immunity, and recommended risk management strategies."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Opinion should address: (1) Section 1983 individual liability vs. municipal liability under Monell doctrine; (2) Qualified immunity analysis - clearly established law and objective reasonableness; (3) Municipal liability requires policy/custom causing violation, not just employee misconduct; (4) Indemnification obligations and insurance coverage; (5) Policy deviation evidence and training implications; (6) Settlement considerations and litigation costs; (7) Prospective risk management including policy updates, training enhancement, and supervision protocols. Emphasize documentation and immediate response importance.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "civil_rights_law"
    },
    "extra_info": {
      "split": "test",
      "index": 215,
      "difficulty": "advanced",
      "court_level": "federal_district",
      "case_citations": ["42 U.S.C. § 1983", "Monell v. Department of Social Services, 436 U.S. 658 (1978)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this medical malpractice case: Emergency room physician Dr. Smith treated patient Williams for chest pain. Dr. Smith ordered basic tests but discharged Williams with diagnosis of muscle strain. Williams suffered massive heart attack 6 hours later. Cardiologist expert testifies that EKG abnormalities warranted cardiac workup. Dr. Smith argues emergency department standards differ from outpatient care standards."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Dr. Smith treated Williams for chest pain in ER, (2) Basic tests performed, diagnosed as muscle strain, (3) Williams discharged, suffered heart attack 6 hours later, (4) Cardiologist expert notes EKG abnormalities requiring cardiac workup, (5) Dr. Smith cites different ER standards. ISSUE: Whether Dr. Smith's diagnostic decisions met emergency medicine standard of care. RULE: Medical malpractice requires breach of applicable standard of care causing injury; emergency physicians held to emergency medicine standards, not specialist standards; expert testimony required for standard of care. APPLICATION: EKG abnormalities suggest warning signs requiring further evaluation; emergency medicine standard focuses on ruling out life-threatening conditions; 6-hour timeframe suggests missed diagnosis was proximate cause. CONCLUSION: Strong malpractice case if expert establishes ER standard required cardiac workup for EKG findings; emergency medicine standard doesn't excuse failure to recognize cardiac warning signs.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "general",
      "legal_domain": "tort_law"
    },
    "extra_info": {
      "split": "test",
      "index": 216,
      "difficulty": "advanced",
      "court_level": "state_trial",
      "case_citations": ["Emergency Medicine Standard of Care Cases"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Examine the precedential impact of Tennessee v. Garner (1985) on police use of deadly force cases. How have subsequent circuit court decisions applied Garner's 'imminent threat' standard, and what analogical reasoning patterns distinguish justified from unjustified shootings in fleeing suspect cases?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Garner established deadly force against fleeing suspects requires imminent threat of death/serious bodily harm to officer or others. Circuit applications show: (1) Immediate threat assessment focuses on weapon possession, violent crime history, and specific threat articulation; (2) Fleeing alone insufficient - 4th, 6th, 9th Circuits require additional danger factors; (3) 5th, 11th Circuits more permissive for violent felony suspects. Distinguishing factors: weapon visibility, verbal threats, prior violence in incident, risk to public safety. Pattern shows evolution toward more restrictive interpretation requiring specific, articulable threat beyond flight from serious crime.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 217,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Tennessee v. Garner, 471 U.S. 1 (1985)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I'm starting a food truck business and need to understand the legal requirements. What licenses and permits do I need? What about health regulations, zoning restrictions, and liability issues? I'm planning to operate in multiple cities within my state."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Food truck operation requires multiple levels of compliance: (1) Business license and tax registration; (2) Food service permits from health departments; (3) Mobile vendor permits from each operating city; (4) Zoning compliance - many cities restrict food truck locations; (5) Fire department permits for propane/cooking equipment; (6) Liability insurance including products liability; (7) Workers' compensation if employees; (8) Vehicle commercial registration and inspection. Multi-city operation complicates compliance as each municipality has different rules. Start with state and county requirements, then research specific city ordinances. Consider consulting attorney for permit compliance strategy.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "business_law"
    },
    "extra_info": {
      "split": "test",
      "index": 218,
      "difficulty": "beginner",
      "court_level": "educational",
      "case_citations": ["Municipal Licensing Requirements"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft advocacy brief arguing that our client's non-compete agreement is unenforceable as an unreasonable restraint of trade. The agreement prohibits any work in the technology industry for 2 years across the entire United States. Client was a junior software developer earning $60,000 annually. Emphasize economic hardship and public policy concerns."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Brief should argue: (1) Geographic scope (entire US) excessive for junior developer role; (2) Two-year duration unreasonable for rapidly evolving technology field; (3) Industry-wide prohibition too broad - should be limited to specific competing activities; (4) No legitimate business interest in preventing junior developer from working anywhere in technology; (5) Economic hardship and public policy favor employee mobility; (6) Blue-pencil doctrine cannot save overly broad restraint; (7) Alternative restrictive covenants (non-solicitation) would protect employer interests without preventing livelihood. Emphasize fundamental right to earn living.",
      "task_type": "opinion_generation",
      "jurisdiction": "general",
      "legal_domain": "employment_law"
    },
    "extra_info": {
      "split": "test",
      "index": 219,
      "difficulty": "intermediate",
      "court_level": "state_trial",
      "case_citations": ["Restatement of Employment Law § 8.06"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC reasoning to this family law custody modification: Father seeks to relocate from California to New York for new job paying 50% more salary. Mother opposes move claiming it will harm 10-year-old daughter's relationship with maternal grandparents who provide daily childcare. Current custody arrangement is 50/50. Father offers increased child support and summer visitation for extended family."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Father seeks CA to NY relocation for 50% salary increase, (2) Mother opposes citing harm to daughter's grandparent relationships, (3) Grandparents provide daily childcare, (4) Current 50/50 custody arrangement, (5) Father offers increased support and summer visitation. ISSUE: Whether relocation serves child's best interests despite family relationship disruption. RULE: Relocation requires best interest analysis weighing economic benefits against relationship disruption; substantial income increase favors move if child benefits; maintaining family relationships important factor. APPLICATION: 50% salary increase provides enhanced opportunities for child; daily grandparent contact disrupted but not eliminated; father's proposal of increased support and summer visits shows mitigation effort; 10-year-old age allows input consideration. CONCLUSION: Relocation likely approved with modified visitation schedule allowing meaningful grandparent contact; economic benefits and mitigation measures support best interest finding.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "California",
      "legal_domain": "family_law"
    },
    "extra_info": {
      "split": "test",
      "index": 220,
      "difficulty": "intermediate",
      "court_level": "family_court",
      "case_citations": ["In re Marriage of LaMusga, 32 Cal. 4th 1072 (2004)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I bought a used car from a dealer who said it had never been in an accident. Three months later, I discovered it was previously declared a total loss due to flood damage. The dealer has a 'sold as-is' sticker on the contract. Do I have any legal recourse despite the 'as-is' clause? What about fraud or misrepresentation claims?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "You likely have strong fraud claims despite 'as-is' language. Key points: (1) 'As-is' clauses don't protect against intentional misrepresentation or fraud; (2) Affirmative false statement about accident history constitutes fraud; (3) Flood damage disclosure often required by state law; (4) Material misrepresentation affects vehicle value and safety; (5) Consumer protection statutes may provide additional remedies and attorney fees. Flood damage is particularly serious affecting safety and value. Document the dealer's statements, obtain vehicle history report, and consider inspection by qualified mechanic. 'As-is' protects against unknown defects, not deliberate concealment of known problems.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "consumer_law"
    },
    "extra_info": {
      "split": "test",
      "index": 221,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["State Consumer Protection Acts"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Analyze the precedential development of the 'state secrets' privilege from United States v. Reynolds (1953) through recent national security cases. How have courts applied Reynolds to balance government secrecy claims against judicial oversight, and what procedural safeguards have emerged?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Reynolds established formal state secrets privilege for protecting military/diplomatic secrets, but subsequent cases reveal evolution: (1) Post-9/11 expansion in al-Haramain and other terrorism cases; (2) 4th Circuit in Sterling shows judicial skepticism of broad claims; (3) D.C. Circuit develops in camera review procedures; (4) 9th Circuit in Mohamed requires specific harm demonstration. Procedural safeguards emerging: ex parte judicial review, segregability analysis, alternative remedies consideration. Pattern shows tension between national security deference and judicial oversight, with courts increasingly demanding specific justification rather than blanket privilege assertions.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "national_security_law"
    },
    "extra_info": {
      "split": "test",
      "index": 222,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["United States v. Reynolds, 345 U.S. 1 (1953)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare a client advisory memo regarding proposed merger transaction between our client (acquiring company) and target company with significant environmental liabilities. Address due diligence requirements, successor liability risks, and structural alternatives to minimize environmental exposure under CERCLA and state environmental laws."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Memo should address: (1) CERCLA successor liability principles and exceptions (Asset Purchase Act, continuity of enterprise, mere continuation doctrines); (2) Comprehensive environmental due diligence including Phase I/II assessments and historical liability review; (3) Transaction structure options - asset purchase vs. stock purchase liability implications; (4) Indemnification and escrow arrangements for environmental costs; (5) Environmental insurance considerations; (6) Regulatory approval requirements and timing; (7) Disclosure obligations to shareholders and regulators. Recommend asset purchase structure with specific excluded liabilities and comprehensive seller indemnification with adequate security.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "environmental_law"
    },
    "extra_info": {
      "split": "test",
      "index": 223,
      "difficulty": "advanced",
      "court_level": "corporate_transaction",
      "case_citations": ["CERCLA 42 U.S.C. § 9607"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this securities law case: CEO Martinez sold $5 million in company stock 2 weeks before announcing poor quarterly earnings that caused 30% stock price decline. SEC investigates for insider trading. Martinez claims sale was pursuant to 10b5-1 plan established 6 months earlier, but investigation reveals plan was modified 1 month before earnings to accelerate sale timing."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) CEO Martinez sold $5M stock 2 weeks before bad earnings announcement, (2) Stock price declined 30% after earnings, (3) Sale claimed under 10b5-1 plan from 6 months prior, (4) Plan modified 1 month before earnings to accelerate timing, (5) SEC investigates insider trading. ISSUE: Whether plan modification undermines good faith defense and constitutes insider trading. RULE: 10b5-1 plans provide affirmative defense if entered in good faith without material nonpublic information; modifications require same good faith standard. APPLICATION: Original plan timing suggests good faith, but modification 1 month before earnings raises questions about access to material information; timing acceleration particularly suspicious; burden on Martinez to prove modification not based on inside information. CONCLUSION: Strong insider trading case due to suspicious timing of plan modification shortly before material adverse information; 10b5-1 defense likely fails due to bad faith modification.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "securities_law"
    },
    "extra_info": {
      "split": "test",
      "index": 224,
      "difficulty": "advanced",
      "court_level": "federal_district",
      "case_citations": ["Rule 10b-5, 17 C.F.R. § 240.10b-5"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "My neighbor's large tree fell on my house during a storm, causing $20,000 in damage. The tree was healthy with no signs of disease, and the storm had 70 mph winds that weren't predicted. My neighbor says it's an 'act of God' and refuses to pay. My insurance company is asking about the neighbor's liability. Who is responsible for storm damage from healthy trees?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Generally, property owners aren't liable for damage from healthy trees that fall during storms. Key legal principles: (1) 'Act of God' defense applies to unforeseeable natural events; (2) Liability typically requires negligence - failure to maintain tree or ignore obvious hazards; (3) Healthy tree with no warning signs usually doesn't create negligence; (4) 70 mph unpredicted winds suggest extraordinary circumstances. Your homeowner's insurance should cover damage under comprehensive coverage. Some states have specific tree damage statutes. If tree had visible disease, damage, or neighbor had notice of hazard, liability might exist. Document tree condition and storm circumstances for insurance claim.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "property_law"
    },
    "extra_info": {
      "split": "test",
      "index": 225,
      "difficulty": "beginner",
      "court_level": "educational",
      "case_citations": ["Nuisance and Negligence Principles"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Compare how federal circuits have interpreted the Americans with Disabilities Act's 'interactive process' requirement in workplace accommodation cases. Analyze the precedential trends in cases where employers claim they engaged in adequate interactive process versus employee claims of superficial compliance."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Circuit interpretations of ADA interactive process show divergent approaches: (1) 1st, 2nd, 3rd Circuits require meaningful bilateral communication with good faith exploration of options; (2) 5th, 7th, 8th Circuits focus on employer's reasonable efforts rather than process outcome; (3) 9th, 11th Circuits emphasize ongoing dialogue and consideration of employee suggestions. Distinguishing factors: timeliness of response, consideration of employee proposals, documentation of alternatives explored, medical information requests. Trend toward requiring substantive engagement rather than perfunctory meetings, with liability for process failures even when accommodation ultimately impossible.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "disability_law"
    },
    "extra_info": {
      "split": "test",
      "index": 226,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Americans with Disabilities Act 42 U.S.C. § 12111"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft a legal opinion for healthcare client addressing medical marijuana program compliance. State law permits medical marijuana, but federal law still prohibits it. Client operates hospital receiving federal funding and wants guidance on employment policies for medical marijuana users and patient treatment protocols."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Opinion should address: (1) Federal vs. state law conflict with Controlled Substances Act superseding state permissions; (2) Federal funding jeopardy under drug-free workplace requirements; (3) Employment policy recommendations including safety-sensitive positions and reasonable accommodation analysis; (4) Patient treatment considerations including pain management alternatives; (5) Professional licensing implications for staff; (6) Risk mitigation strategies including policy documentation and staff training; (7) Monitoring legal developments and federal enforcement priorities. Recommend conservative approach prioritizing federal compliance while staying informed of evolving legal landscape.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "healthcare_law"
    },
    "extra_info": {
      "split": "test",
      "index": 227,
      "difficulty": "advanced",
      "court_level": "regulatory",
      "case_citations": ["Controlled Substances Act 21 U.S.C. § 801"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this products liability case: Child Amy was injured when her bicycle's handlebars suddenly snapped during normal riding, causing her to fall and break her arm. The bike was 2 years old and used regularly. Expert analysis shows metallurgical defect in handlebar welding. Manufacturer BikeCorps argues Amy's parents failed to have annual safety inspections as recommended in owner's manual."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Amy injured when bicycle handlebars snapped during normal use, (2) Bike 2 years old with regular use, (3) Broken arm resulted from fall, (4) Expert shows metallurgical welding defect, (5) Manufacturer cites lack of annual inspections per manual. ISSUE: Whether manufacturing defect creates strict liability despite alleged maintenance failures. RULE: Products liability imposes strict liability for defective products unreasonably dangerous; design and manufacturing defects distinguished from failure to warn; user maintenance duties don't excuse manufacturing defects. APPLICATION: Metallurgical welding defect constitutes manufacturing flaw making product unreasonably dangerous; normal 2-year use shows defect not from misuse; annual inspection requirement doesn't excuse underlying manufacturing defect; child injury during normal use supports liability. CONCLUSION: Strong products liability case for manufacturing defect; inspection recommendations don't bar recovery for inherent welding flaw causing unreasonably dangerous condition.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "general",
      "legal_domain": "tort_law"
    },
    "extra_info": {
      "split": "test",
      "index": 228,
      "difficulty": "intermediate",
      "court_level": "state_trial",
      "case_citations": ["Restatement (Third) of Torts: Products Liability § 2"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I'm getting divorced and my spouse wants alimony. We were married 8 years, both have college degrees, and I earn $120,000 while my spouse earns $45,000. My spouse took 3 years off work to care for our children. What factors determine alimony awards, and what amount and duration should I expect? Are there ways to minimize alimony obligations?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Alimony determination considers multiple factors: (1) Length of marriage (8 years suggests intermediate duration); (2) Income disparity ($75k difference significant); (3) Career sacrifices for childcare (3-year gap impacts earning capacity); (4) Education levels (both college-educated suggests earning potential); (5) Standard of living during marriage; (6) Age and health of parties. Expect temporary rehabilitative alimony to help spouse retrain/re-enter workforce rather than permanent support. Duration might be 2-4 years. Minimization strategies: vocational evaluation showing spouse's earning capacity, evidence of voluntary career choices, settlement negotiations. Consult family law attorney for jurisdiction-specific guidelines and calculation formulas.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "family_law"
    },
    "extra_info": {
      "split": "test",
      "index": 229,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Uniform Marriage and Divorce Act"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Trace the evolution of qualified immunity doctrine from Pierson v. Ray (1967) through recent Supreme Court decisions. Analyze how lower courts have applied the 'clearly established law' standard and identify patterns in cases where immunity is granted versus denied in excessive force contexts."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Qualified immunity evolution shows expansion from Pierson's good faith standard: (1) Harlow (1982) established objective reasonableness test; (2) Anderson (1987) added 'clearly established' requirement; (3) Pearson (2009) allowed order-of-operations flexibility; (4) Recent cases show increasingly restrictive application. Circuit patterns in excessive force: immunity denied when force clearly excessive with obvious alternatives; immunity granted for novel factual scenarios or split-second decisions. 'Clearly established' requires specific precedent with similar facts, not general constitutional principles. Trend toward broader immunity grants despite constitutional violations, creating circuit splits and calls for Supreme Court reconsideration.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "civil_rights_law"
    },
    "extra_info": {
      "split": "test",
      "index": 230,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Pierson v. Ray, 386 U.S. 547 (1967)", "Harlow v. Fitzgerald, 457 U.S. 800 (1982)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare a strategic litigation memo for client facing OSHA workplace safety violations. Proposed penalties total $500,000 for willful violations including fall protection and chemical exposure standards. Client argues violations were due to employee misconduct despite training programs. Analyze defense strategies, settlement possibilities, and potential criminal exposure."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Memo should address: (1) Willful violation elements and defenses - knowledge, intentional disregard, or plain indifference; (2) Employee misconduct defense requirements: adequate training, enforcement, detection efforts; (3) Penalty calculation factors and mitigation arguments; (4) Settlement negotiation strategy including compliance investment; (5) Criminal referral risk under OSH Act willful violations causing death; (6) Repeat violator implications for future penalties; (7) Abatement timeline and ongoing compliance monitoring. Recommend immediate hazard correction, documentation of training/enforcement efforts, and proactive settlement discussions emphasizing safety culture improvements.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "workplace_safety_law"
    },
    "extra_info": {
      "split": "test",
      "index": 231,
      "difficulty": "advanced",
      "court_level": "administrative",
      "case_citations": ["Occupational Safety and Health Act 29 U.S.C. § 651"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this real estate dispute: Buyers entered contract to purchase house for $400,000 with inspection contingency. Home inspection revealed foundation issues requiring $75,000 repairs. Buyers requested seller make repairs or reduce price. Seller refused and claimed buyers must proceed with purchase or forfeit earnest money. Contract states inspection contingency allows withdrawal 'for any reason' within 10 days. Buyers withdrew on day 9."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) $400k purchase contract with inspection contingency, (2) Inspection revealed $75k foundation repairs needed, (3) Buyers requested repairs or price reduction, (4) Seller refused and demanded performance or forfeiture, (5) Contract allows withdrawal 'for any reason' within 10 days, (6) Buyers withdrew on day 9. ISSUE: Whether inspection contingency permits withdrawal without penalty when seller refuses to address discovered issues. RULE: Contingency clauses are strictly construed according to their terms; 'for any reason' language grants broad discretion; timely exercise of contingency rights protects earnest money. APPLICATION: 'For any reason' language clearly permits withdrawal for foundation issues; day 9 withdrawal within 10-day deadline; seller's refusal to negotiate doesn't waive buyers' contingency rights; earnest money protected by proper contingency exercise. CONCLUSION: Buyers entitled to earnest money return due to proper exercise of inspection contingency; seller cannot force performance when contingency properly invoked.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "general",
      "legal_domain": "property_law"
    },
    "extra_info": {
      "split": "test",
      "index": 232,
      "difficulty": "intermediate",
      "court_level": "state_trial",
      "case_citations": ["Real Estate Contract Interpretation"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "My apartment has been without heat for 2 weeks in winter due to broken furnace. Landlord says repair company is backed up and I need to wait. Temperature inside is 45°F. I'm elderly with health conditions. Can I withhold rent, break my lease, or force immediate repairs? What are my rights in habitability situations?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "You have strong habitability rights in this situation: (1) Implied warranty of habitability requires working heating in winter; (2) 45°F temperature constitutes uninhabitable conditions; (3) Options include: rent withholding (check state requirements for notice/escrow), repair and deduct (hire contractor and deduct from rent), constructive eviction (temporary relocation with damages). (4) Health conditions strengthen your position; (5) Document temperature, health impacts, and landlord communications. Many states allow emergency repair provisions for essential services. Consider temporary relocation with landlord responsible for costs. Don't abandon property without legal advice. Contact local tenant rights organization or housing authority for immediate assistance.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "housing_law"
    },
    "extra_info": {
      "split": "test",
      "index": 233,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Implied Warranty of Habitability"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Examine how the Supreme Court's Citizens United v. FEC (2010) decision has been applied by lower courts to corporate political spending restrictions. Analyze the precedential patterns in cases involving disclosure requirements, foreign corporation restrictions, and state-level campaign finance regulations."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Citizens United's impact shows varied lower court applications: (1) Disclosure requirements generally upheld - circuits distinguish between spending restrictions (prohibited) and transparency requirements (permitted); (2) Foreign corporation restrictions survive in 2nd, 4th, D.C. Circuits based on compelling national security interest; (3) State pay-to-play restrictions split circuits - some view as permissible anti-corruption measures, others as impermissible speech restrictions. Pattern shows courts applying Citizens United's core holding (corporate independent expenditure protection) while carving exceptions for compelling interests like corruption prevention and foreign influence. Strict scrutiny applies but informational and anti-corruption interests sometimes sufficient.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "election_law"
    },
    "extra_info": {
      "split": "test",
      "index": 234,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Citizens United v. FEC, 558 U.S. 310 (2010)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft a client alert regarding new cryptocurrency regulations affecting financial institutions. Address compliance requirements for digital asset custody, anti-money laundering obligations, and reporting requirements. Include risk assessment and implementation timeline recommendations for banking clients."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Alert should address: (1) New digital asset custody requirements including segregation, insurance, and recordkeeping obligations; (2) Enhanced BSA/AML compliance for cryptocurrency transactions including suspicious activity monitoring; (3) Reporting requirements for digital asset holdings and transactions; (4) Risk assessment framework for evaluating digital asset services; (5) Implementation timeline with phased compliance approach; (6) Staff training and system upgrade requirements; (7) Regulatory examination expectations and enforcement priorities. Recommend immediate gap analysis, policy updates, and vendor assessment for compliance infrastructure. Emphasize proactive engagement with regulators and industry best practices adoption.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "financial_regulation"
    },
    "extra_info": {
      "split": "test",
      "index": 235,
      "difficulty": "advanced",
      "court_level": "regulatory",
      "case_citations": ["Bank Secrecy Act 31 U.S.C. § 5311"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this bankruptcy case: Debtor filed Chapter 7 bankruptcy listing $200,000 in credit card debt and $50,000 in student loans. Creditor CreditCorp objects to discharge, claiming debtor transferred $100,000 to offshore account 6 months before filing. Debtor claims transfer was legitimate business investment that failed. Trustee seeks to recover transfer as fraudulent conveyance."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Chapter 7 filing with $200k credit card debt, $50k student loans, (2) CreditCorp objects to discharge for fraudulent transfer, (3) $100k offshore transfer 6 months pre-filing, (4) Debtor claims legitimate business investment purpose, (5) Trustee seeks fraudulent conveyance recovery. ISSUE: Whether pre-filing transfer constitutes fraudulent conveyance barring discharge and subject to avoidance. RULE: Fraudulent transfers made with intent to hinder creditors bar discharge; transfers within 2 years presumptively fraudulent absent adequate consideration; trustee may avoid transfers under § 548. APPLICATION: $100k offshore transfer with substantial debt creates badges of fraud; timing 6 months before filing suggests insolvency; business investment claim requires documentation and good faith evidence; lack of consideration supports fraudulent transfer finding. CONCLUSION: Strong case for fraudulent transfer finding; discharge likely denied and transfer avoidable absent compelling business documentation proving legitimate purpose and adequate consideration.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "bankruptcy_law"
    },
    "extra_info": {
      "split": "test",
      "index": 236,
      "difficulty": "advanced",
      "court_level": "federal_bankruptcy",
      "case_citations": ["11 U.S.C. § 548", "11 U.S.C. § 727"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I'm a teacher and my principal is requiring me to remove LGBTQ+ positive books from my classroom library, claiming they violate state education standards. I believe this violates my First Amendment rights and discriminates against LGBTQ+ students. Can the school district restrict classroom materials this way? What are my rights as a public employee?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "This involves complex intersection of educator rights, student rights, and government authority: (1) Public employees have limited First Amendment protection - Garcetti v. Ceballos restricts speech pursuant to official duties; (2) Schools have broad authority over curriculum but cannot discriminate based on viewpoint; (3) Students have right to receive information; (4) Religious/moral objections don't justify content discrimination. Your position depends on: whether books relate to curriculum, state law specifics, educational justification for removal. Consider Equal Protection claims for LGBTQ+ discrimination. Document removal rationale, consult union representative, and review state education standards. Strong case if removal lacks educational justification and appears viewpoint-based.",
      "task_type": "general_chat",
      "jurisdiction": "federal",
      "legal_domain": "education_law"
    },
    "extra_info": {
      "split": "test",
      "index": 237,
      "difficulty": "advanced",
      "court_level": "educational",
      "case_citations": ["Garcetti v. Ceballos, 547 U.S. 410 (2006)", "Board of Education v. Pico, 457 U.S. 853 (1982)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Analyze the precedential impact of Obergefell v. Hodges (2015) on subsequent marriage equality cases involving religious liberty conflicts. How have courts balanced marriage equality rights against religious freedom claims in cases involving wedding vendors, adoption agencies, and employment discrimination?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Obergefell established marriage equality but left religious liberty questions unresolved. Subsequent cases show pattern: (1) Public accommodations laws generally upheld against religious vendors (Masterpiece Cakeshop shows narrow exception for hostility); (2) Religious adoption agencies receive mixed results - some circuits require equal treatment, others allow religious exemptions; (3) Employment discrimination increasingly protects LGBTQ+ workers under Title VII per Bostock. Balancing factors: government compelling interest, neutral application of laws, religious organization vs. public accommodation distinction. Trend favors marriage equality in public sphere while protecting religious organization autonomy in internal matters.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "constitutional_law"
    },
    "extra_info": {
      "split": "test",
      "index": 238,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Obergefell v. Hodges, 576 U.S. 644 (2015)", "Masterpiece Cakeshop v. Colorado Civil Rights Comm'n, 138 S. Ct. 1719 (2018)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare a compliance advisory for technology client regarding new AI governance regulations. Address algorithmic bias testing requirements, transparency obligations for automated decision-making, and data protection compliance for AI training datasets. Include implementation roadmap and ongoing monitoring requirements."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Advisory should cover: (1) Algorithmic bias testing methodologies and documentation requirements for protected class impacts; (2) Transparency obligations including automated decision-making disclosures and explanation rights; (3) Data protection compliance for training datasets including consent, minimization, and retention limits; (4) Risk assessment frameworks for high-risk AI applications; (5) Implementation roadmap with testing phases, policy development, and training requirements; (6) Ongoing monitoring including performance metrics, bias detection, and incident response; (7) Regulatory examination preparation and documentation standards. Recommend cross-functional governance committee and phased compliance approach prioritizing high-risk applications.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "technology_law"
    },
    "extra_info": {
      "split": "test",
      "index": 239,
      "difficulty": "advanced",
      "court_level": "regulatory",
      "case_citations": ["Emerging AI Governance Frameworks"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this immigration case: Maria entered U.S. illegally 20 years ago, established business employing 50 people, has U.S. citizen children, and no criminal record. ICE initiated removal proceedings after workplace raid. Maria applies for cancellation of removal claiming exceptional hardship to community and family. Government argues immigration violations warrant removal regardless of community ties."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Maria illegally entered 20 years ago, (2) Established business employing 50 people, (3) U.S. citizen children, (4) No criminal record, (5) ICE removal proceedings after workplace raid, (6) Seeks cancellation for exceptional hardship to community/family. ISSUE: Whether community impact and family ties constitute exceptional hardship justifying cancellation of removal. RULE: Cancellation of removal for non-LPRs requires: (1) 10+ years presence, (2) good moral character, (3) exceptional hardship to qualifying U.S. relatives. APPLICATION: 20-year presence satisfies duration; business ownership and no criminal record support good character; U.S. citizen children qualify as relatives; community employment impact and family separation constitute exceptional hardship. CONCLUSION: Strong case for cancellation due to lengthy presence, community ties, and exceptional hardship to citizen children; business impact and family integration support discretionary relief.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "immigration_law"
    },
    "extra_info": {
      "split": "test",
      "index": 240,
      "difficulty": "advanced",
      "court_level": "immigration_court",
      "case_citations": ["8 U.S.C. § 1229b(b)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I cosigned a car loan for my adult child who has stopped making payments. The lender is demanding I pay the full $25,000 balance immediately. I thought cosigning just meant I'd help if my child couldn't pay, not that I'd be fully responsible. What are my legal obligations as a cosigner, and do I have any defenses?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "As cosigner, you're fully liable for the debt - not just a backup payer. Key points: (1) Cosigners have primary liability equal to main borrower; (2) Lender can pursue you without first pursuing your child; (3) Default triggers full debt acceleration; (4) Potential defenses: lack of proper cosigner disclosure, fraud in inducement, lender's failure to follow collection procedures. Options include: negotiating payment plan, asking lender to work with your child first, or asserting any disclosure violations. Unfortunately, 'I thought I was just helping' isn't a legal defense. Review loan documents for cosigner disclosure requirements and consider whether lender provided required FTC cosigner notice. Consult attorney for specific defense analysis.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "consumer_law"
    },
    "extra_info": {
      "split": "test",
      "index": 241,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["FTC Cosigner Notice Requirements"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Compare how different federal circuits have interpreted the Computer Fraud and Abuse Act's 'exceeds authorized access' provision in employment context cases. Analyze the precedential split between broad and narrow interpretations and identify factors courts use to distinguish permissible from criminal computer use."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "CFAA 'exceeds authorized access' interpretations create significant circuit split: (1) Broad interpretation (1st, 5th, 7th, 11th Circuits): violation of computer use policies constitutes exceeding authorization; employee misuse triggers liability. (2) Narrow interpretation (2nd, 4th, 9th Circuits): requires circumvention of technical access controls; policy violations alone insufficient. (3) 6th Circuit takes middle approach considering purpose and method. Distinguishing factors: technical barriers vs. policy restrictions, data type accessed, extraction methods, intended use. Van Buren (2021) Supreme Court decision favors narrow interpretation, limiting CFAA to gate-keeping mechanisms rather than policy enforcement.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "cyber_law"
    },
    "extra_info": {
      "split": "test",
      "index": 242,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Computer Fraud and Abuse Act 18 U.S.C. § 1030", "Van Buren v. United States, 141 S. Ct. 1648 (2021)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Draft a risk management memo for pharmaceutical client facing multiple product liability lawsuits over diabetes medication allegedly causing kidney damage. Address litigation strategy, regulatory compliance issues, and potential class action consolidation risks. Include recommendations for ongoing clinical trial management and FDA communication."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Memo should address: (1) Litigation defense strategy including causation challenges, scientific evidence evaluation, and expert witness coordination; (2) MDL consolidation risks and venue selection implications; (3) FDA reporting obligations for adverse events and label update requirements; (4) Clinical trial protocol modifications and safety monitoring enhancements; (5) Insurance coverage coordination and settlement authority; (6) Public relations and stakeholder communication strategy; (7) Regulatory compliance including REMS programs and post-market surveillance. Recommend proactive FDA engagement, comprehensive discovery preservation, and coordinated defense strategy across jurisdictions while maintaining ongoing regulatory compliance and safety monitoring.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "pharmaceutical_law"
    },
    "extra_info": {
      "split": "test",
      "index": 243,
      "difficulty": "advanced",
      "court_level": "multidistrict_litigation",
      "case_citations": ["FDA Adverse Event Reporting Requirements"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this construction contract dispute: General contractor BuildCorp subcontracted electrical work to ElectricCo for $200,000. Midway through project, steel prices increased 40% due to supply chain issues. ElectricCo demands additional $50,000 or threatens to abandon project. BuildCorp refuses, claiming fixed-price contract. ElectricCo stops work, causing project delays and forcing BuildCorp to hire replacement contractor for $300,000."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) $200k electrical subcontract between BuildCorp and ElectricCo, (2) 40% steel price increase due supply chain issues, (3) ElectricCo demands $50k additional or abandonment threat, (4) BuildCorp refuses citing fixed-price terms, (5) ElectricCo stops work causing delays, (6) Replacement contractor costs $300k. ISSUE: Whether material cost increases excuse performance or constitute breach of fixed-price contract. RULE: Fixed-price contracts generally allocate market risk to contractor; impossibility/impracticability defense requires extreme and unforeseeable circumstances; breach triggers damages for cover costs and delays. APPLICATION: 40% price increase significant but markets inherently volatile; COVID supply chain issues arguably foreseeable business risk; fixed-price contract language shows risk allocation; abandonment constitutes material breach triggering cover damages. CONCLUSION: ElectricCo likely liable for breach; BuildCorp entitled to excess completion costs ($100k) plus delay damages; impracticability defense unlikely to succeed absent extreme circumstances.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "general",
      "legal_domain": "construction_law"
    },
    "extra_info": {
      "split": "test",
      "index": 244,
      "difficulty": "advanced",
      "court_level": "state_trial",
      "case_citations": ["Restatement (Second) of Contracts § 261"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "My HOA is trying to fine me $500 for painting my front door blue, claiming it violates the 'neutral color' requirement in the CC&Rs. Other homes have colorful doors that haven't been cited. I think this is selective enforcement and the rule is too vague. Can HOAs enforce subjective aesthetic rules, and do I have any defenses for selective enforcement?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "You have several potential defenses: (1) Selective enforcement - if similar violations aren't cited, creates arbitrary application argument; (2) Vagueness challenge - 'neutral color' may be too subjective to enforce fairly; (3) Due process concerns - HOA must follow proper notice and hearing procedures. HOAs can enforce aesthetic rules if: properly authorized in CC&Rs, reasonably applied, and procedurally fair. Document other colorful doors with photos, review CC&R language carefully, and request HOA enforcement records. Challenge vague terms and demand consistent application. Consider requesting architectural committee review or mediation. Business judgment rule protects reasonable HOA decisions, but arbitrary enforcement and vague rules are vulnerable to challenge.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "property_law"
    },
    "extra_info": {
      "split": "test",
      "index": 245,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["HOA Enforcement and Due Process Requirements"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Analyze how courts have applied the Supreme Court's decision in Janus v. AFSCME (2018) to subsequent public sector union cases. Compare circuit court interpretations regarding agency fees, union security arrangements, and First Amendment boundaries in public employment contexts."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Janus eliminated mandatory agency fees for public sector unions, establishing First Amendment protection for non-members. Circuit applications show: (1) 7th, 9th Circuits strictly apply Janus to invalidate all fee arrangements; (2) 2nd, 4th Circuits narrow Janus to mandatory fees, allowing voluntary arrangements; (3) D.C. Circuit addresses union access to employee contact information. Key issues: retroactive fee refunds (circuits split), union exclusive representation obligations, member-only contract clauses. Pattern shows broad First Amendment protection for public employees against compelled speech while maintaining union representation structures. Courts distinguish between prohibited compulsion and permissible voluntary association.",
      "task_type": "precedent_analysis",
      "jurisdiction": "federal",
      "legal_domain": "labor_law"
    },
    "extra_info": {
      "split": "test",
      "index": 246,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Janus v. AFSCME, 138 S. Ct. 2448 (2018)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Prepare regulatory compliance assessment for energy client regarding new carbon emission standards. Address implementation requirements, monitoring and reporting obligations, and enforcement mechanisms. Include cost-benefit analysis framework and recommendation for compliance strategy given pending federal legislation."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Assessment should cover: (1) Implementation timeline and technical requirements for emission reduction technologies; (2) Continuous monitoring systems and quarterly reporting protocols; (3) Enforcement mechanisms including penalties, shutdown authority, and permit revocation; (4) Cost-benefit analysis including technology costs, operational impacts, and environmental benefits; (5) Compliance strategy options including early implementation credits, emission trading participation, and technology partnerships; (6) Federal legislation monitoring and adaptation planning; (7) Stakeholder engagement including community input and environmental justice considerations. Recommend phased implementation approach with early compliance benefits while monitoring federal developments for potential preemption or modification.",
      "task_type": "opinion_generation",
      "jurisdiction": "federal",
      "legal_domain": "environmental_law"
    },
    "extra_info": {
      "split": "test",
      "index": 247,
      "difficulty": "advanced",
      "court_level": "regulatory",
      "case_citations": ["Clean Air Act 42 U.S.C. § 7401"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Use FIRAC to analyze this tax evasion case: Business owner Johnson failed to report $500,000 in cash receipts over 3 years, filed false tax returns showing losses, and maintained offshore accounts. IRS criminal investigation discovered pattern of cash transactions and unreported income. Johnson claims poor recordkeeping and reliance on incompetent accountant. Government seeks criminal prosecution for tax evasion."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Johnson failed to report $500k cash receipts over 3 years, (2) Filed false returns showing losses despite income, (3) Maintained offshore accounts, (4) Pattern of unreported cash transactions, (5) Claims poor recordkeeping and accountant reliance, (6) Government seeks criminal tax evasion prosecution. ISSUE: Whether evidence establishes willful tax evasion beyond reasonable doubt. RULE: Criminal tax evasion requires: (1) tax deficiency, (2) willful attempt to evade, (3) affirmative act of evasion; good faith reliance on professional may negate willfulness. APPLICATION: $500k unreported income establishes substantial deficiency; false loss returns and offshore accounts show affirmative evasion acts; 3-year pattern suggests willful conduct; accountant reliance defense undermined by false information provided. CONCLUSION: Strong criminal case for tax evasion; pattern of conduct and false returns evidence willfulness; accountant reliance defense likely insufficient given deliberate concealment of income sources.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "tax_law"
    },
    "extra_info": {
      "split": "test",
      "index": 248,
      "difficulty": "advanced",
      "court_level": "federal_district",
      "case_citations": ["26 U.S.C. § 7201"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "I'm a freelance photographer and a client is refusing to pay my $5,000 invoice for wedding photography services. They claim some photos were blurry and don't meet professional standards. My contract says payment is due regardless of client satisfaction, but they're threatening to sue for breach of contract. What are my options for collecting payment, and how strong is their potential counterclaim?"
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "Your collection options include: (1) Demand letter with payment deadline; (2) Small claims court if within limits; (3) Civil lawsuit for breach of contract; (4) Mechanics lien if state law permits for services. Their counterclaim strength depends on: contract terms regarding quality standards, objective evidence of professional standards breach, substantial performance doctrine. 'Regardless of satisfaction' clause helps but doesn't excuse material performance failures. Document evidence: contract terms, delivered photos, industry standards, client communications. If photos substantially conform to reasonable professional standards, strong collection case. Consider settlement negotiation for partial payment to avoid litigation costs. Review contract for attorney fee provisions and dispute resolution clauses.",
      "task_type": "general_chat",
      "jurisdiction": "general",
      "legal_domain": "contract_law"
    },
    "extra_info": {
      "split": "test",
      "index": 249,
      "difficulty": "intermediate",
      "court_level": "educational",
      "case_citations": ["Professional Service Contract Law"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  },
  {
    "data_source": "legal_reasoning",
    "prompt": [
      {
        "role": "user",
        "content": "Apply FIRAC analysis to this criminal appeal: Defendant convicted of armed robbery based primarily on eyewitness identification from victim who viewed suspect under poor lighting conditions during nighttime attack. Defense challenges conviction arguing unreliable identification violated due process. Prosecution argues corroborating evidence including defendant's presence in area and similar weapon recovery supports conviction."
      }
    ],
    "reward_model": {
      "style": "multi_task_legal",
      "ground_truth": "FACTS: (1) Armed robbery conviction based primarily on eyewitness identification, (2) Poor lighting conditions during nighttime attack, (3) Defense challenges identification reliability, (4) Due process violation claimed, (5) Corroborating evidence: defendant's area presence and similar weapon. ISSUE: Whether unreliable eyewitness identification under poor conditions violates due process despite corroborating evidence. RULE: Due process requires exclusion of identification evidence when reliability so poor that admission violates fundamental fairness; corroborating evidence may cure identification defects. APPLICATION: Poor lighting and nighttime conditions create reliability concerns; single witness identification without independent verification problematic; area presence and weapon evidence provide some corroboration but don't cure initial identification issues; totality of circumstances analysis required. CONCLUSION: Conviction vulnerable on appeal if identification procedures fundamentally unreliable; strength of corroborating evidence determines whether due process violated; remand for reliability hearing likely unless corroboration clearly sufficient.",
      "task_type": "judicial_reasoning",
      "jurisdiction": "federal",
      "legal_domain": "criminal_law"
    },
    "extra_info": {
      "split": "test",
      "index": 250,
      "difficulty": "advanced",
      "court_level": "federal_circuit",
      "case_citations": ["Manson v. Brathwaite, 432 U.S. 98 (1977)"],
      "tools_available": ["courtlistener", "brave_search"]
    }
  }
]

