# Arabic Morphology Guide for AraStudy

## 1. Root-Pattern System
Root carries core semantics; patterns create surface word forms.

Example root: **كتب**
- كتب (verb)
- كتاب (noun)
- كاتب (active participle)
- مكتوب (passive participle)
- مكتبة (place noun)
- كتابة (verbal noun)
- كتب (broken plural)
- كتاب (agentive/broken plural family)

## 2. Common Patterns (reference)
| Pattern | Typical function | Example |
|---|---|---|
| فعل | base verb | كتب |
| فاعل | active participle | كاتب |
| مفعول | passive participle | مكتوب |
| فعال | noun | كتاب |
| فعالة | action noun | كتابة |
| مفعلة | place noun | مكتبة |
| فعيل | adjective | كبير |
| أفعل | comparative | أكبر |
| تفعيل | verbal noun | تعليم |
| افتعال | verbal noun VIII | اجتماع |

## 3. Edge Cases
### 3.1 Weak roots
- قال (قول), نام (نوم), مشى (مشي)

### 3.2 Broken plurals
- كتاب → كتب
- عالم → علماء
- مدرسة → مدارس

### 3.3 Quadriliteral roots
- زلزل
- ترجم

## 4. 100 Test Words (seed list)
| # | Word | Root | Pattern | Category |
|---|---|---|---|---|
| 1 | كتب | كتب | فعل | regular |
| 2 | كاتب | كتب | فاعل | regular |
| 3 | مكتوب | كتب | مفعول | regular |
| 4 | كتاب | كتب | فعال | regular |
| 5 | كتابة | كتب | فعالة | regular |
| 6 | مكتبة | كتب | مفعلة | regular |
| 7 | يكتب | كتب | يفعل | regular |
| 8 | كتبت | كتب | فعلت | regular |
| 9 | كتابي | كتب | فعالي | regular |
| 10 | كتّاب | كتب | فعال | broken_plural |
| 11 | علم | علم | فعل | regular |
| 12 | عالم | علم | فاعل | regular |
| 13 | معلوم | علم | مفعول | regular |
| 14 | تعليم | علم | تفعيل | regular |
| 15 | معلم | علم | مفعل | regular |
| 16 | معلمة | علم | مفعلة | regular |
| 17 | علوم | علم | فعول | broken_plural |
| 18 | علماء | علم | فعلاء | broken_plural |
| 19 | يتعلم | علم | يتفعل | regular |
| 20 | تعلم | علم | تفعل | regular |
| 21 | درس | درس | فعل | regular |
| 22 | دارس | درس | فاعل | regular |
| 23 | مدروس | درس | مفعول | regular |
| 24 | دراسة | درس | فعالة | regular |
| 25 | مدرسة | درس | مفعلة | regular |
| 26 | مدارس | درس | مفاعل | broken_plural |
| 27 | دروس | درس | فعول | broken_plural |
| 28 | يدرس | درس | يفعل | regular |
| 29 | درسنا | درس | فعلنا | regular |
| 30 | مدرسي | درس | مفعلي | regular |
| 31 | عمل | عمل | فعل | regular |
| 32 | عامل | عمل | فاعل | regular |
| 33 | معمول | عمل | مفعول | regular |
| 34 | أعمال | عمل | أفعال | broken_plural |
| 35 | معمل | عمل | مفعل | regular |
| 36 | عمال | عمل | فعال | broken_plural |
| 37 | يعمل | عمل | يفعل | regular |
| 38 | عملي | عمل | فعلي | regular |
| 39 | عملية | عمل | فعلية | regular |
| 40 | تعامل | عمل | تفاعل | regular |
| 41 | خرج | خرج | فعل | regular |
| 42 | خارج | خرج | فاعل | regular |
| 43 | خروج | خرج | فعول | regular |
| 44 | مخرج | خرج | مفعل | regular |
| 45 | استخراج | خرج | استفعال | regular |
| 46 | استخرج | خرج | استفعل | regular |
| 47 | يخرج | خرج | يفعل | regular |
| 48 | خرجوا | خرج | فعلوا | regular |
| 49 | خارجي | خرج | فاعلي | regular |
| 50 | مخارج | خرج | مفاعل | broken_plural |
| 51 | دخل | دخل | فعل | regular |
| 52 | داخل | دخل | فاعل | regular |
| 53 | دخول | دخل | فعول | regular |
| 54 | مدخل | دخل | مفعل | regular |
| 55 | تدخل | دخل | تفعل | regular |
| 56 | أدخل | دخل | أفعل | regular |
| 57 | يدخل | دخل | يفعل | regular |
| 58 | داخلية | دخل | فاعلية | regular |
| 59 | مداخل | دخل | مفاعل | broken_plural |
| 60 | دخيل | دخل | فعيل | regular |
| 61 | قال | قول | فعل | weak |
| 62 | يقول | قول | يفعل | weak |
| 63 | قول | قول | فعل | weak |
| 64 | قائل | قول | فاعل | weak |
| 65 | مقول | قول | مفعول | weak |
| 66 | اقوال | قول | أفعال | weak |
| 67 | مقالة | قول | مفعلة | weak |
| 68 | قيل | قول | فعل | weak |
| 69 | نقول | قول | نفعل | weak |
| 70 | يقولون | قول | يفعلون | weak |
| 71 | نام | نوم | فعل | weak |
| 72 | ينام | نوم | يفعل | weak |
| 73 | نوم | نوم | فعل | weak |
| 74 | نائم | نوم | فاعل | weak |
| 75 | منام | نوم | مفعل | weak |
| 76 | منامة | نوم | مفعلة | weak |
| 77 | ناموا | نوم | فعلوا | weak |
| 78 | تنام | نوم | تفعل | weak |
| 79 | نيام | نوم | فعال | weak |
| 80 | منوم | نوم | مفعول | weak |
| 81 | مشى | مشي | فعل | weak |
| 82 | يمشي | مشي | يفعل | weak |
| 83 | مشي | مشي | فعل | weak |
| 84 | ماش | مشي | فاع | weak |
| 85 | ممشى | مشي | مفعل | weak |
| 86 | مشاة | مشي | فعاله | weak |
| 87 | مشوا | مشي | فعلوا | weak |
| 88 | تمشي | مشي | تفعل | weak |
| 89 | مشاية | مشي | فعالة | weak |
| 90 | ممشاة | مشي | مفعلة | weak |
| 91 | زلزل | زلزل | فعلل | quadriliteral |
| 92 | زلزلة | زلزل | فعللة | quadriliteral |
| 93 | مترجم | ترجم | مفعل | quadriliteral |
| 94 | ترجمة | ترجم | فعلة | quadriliteral |
| 95 | برمج | برمج | فعلل | quadriliteral |
| 96 | برمجة | برمج | فعللة | quadriliteral |
| 97 | في | - | - | function |
| 98 | من | - | - | function |
| 99 | على | - | - | function |
| 100 | الى | - | - | function |

Use this guide as:
- Pre-RQ lexical reference
- RQ0 probing source
- Paper §2 examples seed
- Analyzer sanity-check list
