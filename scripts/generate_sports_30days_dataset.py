import os
import sys
import random
import re
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure seed reproducibility
random.seed(42)

OUTPUT_DIR = "data/01_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = datetime(2026, 8, 16, 8, 0, 0)
DAYS_COUNT = 30

# 100% English Sports Thematic Narrative Arcs (Football, Tennis, F1, Basketball, Scandals & Drama)
TOPIC_ARCS = [
    {
        "id": "epl_kickoff_chelsea_mancity",
        "name": "Premier League Kickoff & Chelsea vs Man City",
        "sport": "football",
        "peak_day": 2, # Aug 18
        "duration": 5,
        "is_drama": True,
        "base_sentiment": 0.1,
        "controversy": 0.45,
        "hashtags": ["epl", "premierleague", "mancity", "chelsea", "haaland", "football"],
        "templates": [
            ("facebook", "fanpage", "Stamford Bridge clash: Manchester City defeat Chelsea 2-0 on the opening weekend. Erling Haaland marks his 100th appearance with his 91st goal for the club. Can anyone stop Pep Guardiola's machine this season? #premierleague #mancity #haaland"),
            ("facebook", "fanpage", "Controversy at Stamford Bridge: Chelsea fans furious as Enzo Fernandez penalty claim is waived off after VAR check. Was it a clear handball? Give your thoughts below. #chelsea #premierleague #var"),
            ("tiktok", "short_video", "Erling Haaland muscles past Cucurella and chips Robert Sanchez to open the scoring! Absolutely clinical finish #haaland #mancity #chelsea #football #fyp"),
            ("tiktok", "short_video", "POV: Chelsea spending £300M on midfielders only for Mateo Kovacic to solo dribble through them and score a worldie 😭 #chelsea #mancity #epl #football"),
            ("youtube", "official_channel", "EXTENDED HIGHLIGHTS: CHELSEA 0-2 MANCHESTER CITY | HAALAND 100TH GAME GOAL & KOVACIC STUNNER | PREMIER LEAGUE"),
            ("youtube", "channel", "Tactical Breakdown: Why Enzo Maresca's tactical setup left Chelsea completely vulnerable against Pep Guardiola's press"),
            ("reddit", "subreddit", "[Post-Match Thread] Chelsea 0-2 Manchester City (Premier League Matchweek 1)"),
            ("reddit", "subreddit", "Enzo Fernandez handball penalty appeal denied by VAR review: Discussion and multiple camera angles"),
            ("news", "sports_news", "Premier League: Erling Haaland scores on 100th appearance as Manchester City outclass Chelsea at Stamford Bridge"),
            ("news", "sports_news", "Chelsea 0-2 Man City: Mateo Kovacic scores stunning solo goal against former club to seal opening weekend victory")
        ],
        "variations": [
            "What a clinical display from the reigning champions at the Bridge.",
            "Chelsea showed flashes of promise under Maresca, but defensive lapses proved costly once again.",
            "Mateo Kovacic refused to celebrate against his former club, but that strike was pure class.",
            "Haaland proving all critics wrong on day one with a trademark power finish.",
            "Referee Anthony Taylor and VAR under scrutiny over the non-call on the handball appeal."
        ]
    },
    {
        "id": "sinner_doping_scandal",
        "name": "Jannik Sinner Doping Controversy & ITIA Ruling",
        "sport": "tennis",
        "peak_day": 5, # Aug 20-21
        "duration": 10,
        "is_drama": True,
        "base_sentiment": -0.4,
        "controversy": 0.88,
        "hashtags": ["tennis", "sinner", "doping", "usopen", "itia", "kyrgios"],
        "templates": [
            ("facebook", "fanpage", "BREAKING SCANDAL: World No. 1 Jannik Sinner tested positive twice for banned steroid Clostebol in March but avoids ban! The ITIA accepted his defense of accidental contamination via his physio. Massive uproar across the tennis world! #tennis #sinner #doping #usopen"),
            ("facebook", "fanpage", "Nick Kyrgios erupts on social media over Sinner verdict: 'Ridiculous - whether it was accidental or planned. You get tested twice with a banned substance... you should be gone for 2 years!'. Do you agree with Kyrgios? #tennis #kyrgios #sinner"),
            ("tiktok", "short_video", "Everything you need to know about the Jannik Sinner doping scandal that shocked the sports world just days before the US Open #sinner #tennis #usopen #drama"),
            ("tiktok", "short_video", "Tennis stars react to Jannik Sinner escaping doping suspension: Nick Kyrgios and Denis Shapovalov call out double standards! #tennis #kyrgios #sinnerdoping"),
            ("youtube", "official_channel", "JANNIK SINNER DOPING SCANDAL EXPLAINED: Why the World No. 1 Escaped a Suspension | In-Depth Legal Breakdown"),
            ("youtube", "channel", "Double Standards in Tennis? Investigating the ITIA 33-Page Independent Tribunal Report on Jannik Sinner"),
            ("reddit", "subreddit", "[Official Release] ITIA concludes Jannik Sinner bore 'No Fault or Negligence' for two adverse analytical findings of Clostebol"),
            ("reddit", "subreddit", "Denis Shapovalov on Sinner case: 'Different rules for different players. Can't imagine what every other player who got banned feels right now.'"),
            ("news", "sports_news", "Jannik Sinner cleared of doping fault after twice testing positive for banned anabolic steroid clostebol"),
            ("news", "sports_news", "Tennis world divided as World No. 1 Jannik Sinner avoids suspension following contamination ruling ahead of US Open")
        ],
        "variations": [
            "The timing of this announcement right before the US Open draw has caused major suspicion among players.",
            "Many lower-ranked players point out that they faced immediate provisional bans under identical circumstances.",
            "Sinner fired his physio and fitness coach following the independent tribunal ruling.",
            "WADA still holds the right to appeal the tribunal decision to the Court of Arbitration for Sport (CAS).",
            "A dark cloud hanging over the men's draw at Flushing Meadows as press conferences turn fiery."
        ]
    },
    {
        "id": "transfer_deadline_sancho_sterling_osimhen",
        "name": "Transfer Deadline Day: Sancho, Sterling, Osimhen Sagas",
        "sport": "football",
        "peak_day": 15, # Aug 30-31
        "duration": 7,
        "is_drama": True,
        "base_sentiment": 0.2,
        "controversy": 0.65,
        "hashtags": ["transfer", "deadlineday", "sancho", "sterling", "chelsea", "arsenal", "herewego"],
        "templates": [
            ("facebook", "fanpage", "HERE WE GO! Jadon Sancho completes dramatic deadline day move from Manchester United to Chelsea on loan with obligation to buy for £25m! Thoughts on this deal? #sancho #chelsea #mufc #transfer"),
            ("facebook", "fanpage", "Shock London switch: Raheem Sterling joins Arsenal on season-long loan after being exiled to Chelsea's 'bomb squad'. The winger took a major wage cut to join Arteta's title challenge! #sterling #arsenal #chelsea"),
            ("facebook", "fanpage", "Transfer collapse of the summer: Victor Osimhen's proposed move to Chelsea collapses in the final hours over wage negotiations. Striker frozen out at Napoli! #osimhen #chelsea #transfers"),
            ("tiktok", "short_video", "Fabrizio Romano working 48 hours straight without sleep on Transfer Deadline Day: Sancho to Chelsea and Sterling to Arsenal confirmed! #fabrizioromano #transfer #deadlineday #football"),
            ("tiktok", "short_video", "Manchester United fans celebrating getting rid of Sancho only to realize Chelsea got him on a bargain obligation clause 😭 #manutd #sancho #chelsea #football"),
            ("youtube", "official_channel", "DEADLINE DAY LIVE: SANCHO TO CHELSEA, STERLING TO ARSENAL, UGARTE TO MANCHESTER UNITED OFFICIALLY CONFIRMED!"),
            ("youtube", "channel", "Transfer Window Winners and Losers 2026: Analyzing Chelsea's 40-man squad and Arsenal's title depth"),
            ("reddit", "subreddit", "[Fabrizio Romano] Jadon Sancho to Chelsea, HERE WE GO! Loan deal with obligation to buy clause around £25m"),
            ("reddit", "subreddit", "[Official] Raheem Sterling joins Arsenal on season-long loan from Chelsea with no purchase option"),
            ("news", "sports_news", "Transfer Deadline Day Round-Up: Chelsea seal Jadon Sancho deal, Arsenal sign Sterling, Manuel Ugarte joins Man Utd"),
            ("news", "sports_news", "Victor Osimhen stranded at Napoli after blockbuster Chelsea deadline day move falls apart over wage demands")
        ],
        "variations": [
            "One of the craziest transfer deadline days in recent memory with multiple top-six London swaps.",
            "Mikel Arteta believes Raheem Sterling can provide the decisive edge in the Premier League title race against Man City.",
            "Chelsea's squad building strategy continues to baffle pundits across European football.",
            "Manuel Ugarte finally completes his medical to reinforce Erik ten Hag's midfield options at Old Trafford.",
            "Osimhen now facing the prospect of training with Napoli reserves until the January window opens."
        ]
    },
    {
        "id": "manutd_0_3_liverpool_derby_tenhag_crisis",
        "name": "Man United 0-3 Liverpool Derby & Ten Hag Crisis",
        "sport": "football",
        "peak_day": 16, # Sep 1
        "duration": 6,
        "is_drama": True,
        "base_sentiment": -0.65,
        "controversy": 0.78,
        "hashtags": ["manutd", "liverpool", "tenhag", "casemiro", "salah", "oldtrafford", "football"],
        "templates": [
            ("facebook", "fanpage", "OLD TRAFFORD HUMILIATION: Manchester United dismantled 0-3 by Liverpool! Casemiro commits two catastrophic first-half errors before being hooked at halftime! #manutd #liverpool #casemiro"),
            ("facebook", "fanpage", "Mohamed Salah drops a bombshell post-match interview: 'Nobody in the club has talked to me about a new contract yet, so this might be my last season at Old Trafford.' Unbelievable drama! #salah #liverpool"),
            ("facebook", "fanpage", "Intense pressure mounting on Erik ten Hag: After spending £600M, Manchester United are totally outplayed and outfought by Arne Slot's Liverpool. Time for INEOS to make a decision? #tenhagout #manutd"),
            ("tiktok", "short_video", "Casemiro's horror 45 minutes vs Liverpool that had Ten Hag substituting him at halftime for a 20-year-old rookie #casemiro #manutd #liverpool #premierleague #fyp"),
            ("tiktok", "short_video", "Thousands of Manchester United fans walking out of Old Trafford in the 70th minute after Salah makes it 3-0 😭 #manunited #liverpool #oldtrafford"),
            ("youtube", "official_channel", "MANCHESTER UNITED 0-3 LIVERPOOL | LUIS DIAZ BRACE & SALAH MASTERCLASS DISMANTLE TEN HAG AT OLD TRAFFORD"),
            ("youtube", "channel", "Roy Keane & Gary Neville on Sky Sports: 'Manchester United were totally carved open. Casemiro's legs have gone at the top level'"),
            ("reddit", "subreddit", "[Post-Match Thread] Manchester United 0-3 Liverpool (Premier League)"),
            ("reddit", "subreddit", "Jamie Carragher on Casemiro: 'Leave the football before the football leaves you.' Honest assessment or disrespect to a 5-time Champions League winner?"),
            ("news", "sports_news", "Manchester United 0-3 Liverpool: Luis Diaz double and Mohamed Salah strike pile immense pressure on Erik ten Hag"),
            ("news", "sports_news", "Mohamed Salah hints at Liverpool departure following dominant Old Trafford masterclass")
        ],
        "variations": [
            "Arne Slot becomes the first Liverpool manager since Bob Paisley to win his first visit to Old Trafford.",
            "Erik ten Hag insists in his post-match press conference that his team will compete for trophies by May.",
            "Manchester United have now lost two of their first three Premier League matches to start the campaign.",
            "Luis Diaz looks reborn under Slot's high-pressing system with three goals in two matches.",
            "The midfield disparity between Ryan Gravenberch and Casemiro was the deciding factor throughout the 90 minutes."
        ]
    },
    {
        "id": "usopen_shocks_alcaraz_djokovic_exit",
        "name": "US Open Shock Upsets: Alcaraz & Djokovic Eliminated",
        "sport": "tennis",
        "peak_day": 14, # Aug 30
        "duration": 5,
        "is_drama": True,
        "base_sentiment": -0.3,
        "controversy": 0.55,
        "hashtags": ["usopen", "alcaraz", "djokovic", "tennis", "upset", "grandslam"],
        "templates": [
            ("facebook", "fanpage", "COLOSSAL UPSET AT US OPEN: World No. 3 and French Open/Wimbledon champion Carlos Alcaraz crashed out in straight sets (6-1, 7-5, 6-4) to unseeded Dutchman Botic van de Zandschulp! One of the biggest shocks in Grand Slam history! #alcaraz #usopen #tennis"),
            ("facebook", "fanpage", "Less than 24 hours after Alcaraz was stunned, defending champion Novak Djokovic is knocked out in the 3rd round by Alexei Popyrin! For the first time since 2017, a Grand Slam enters the second week without Djokovic, Nadal, or Federer! #djokovic #usopen #tennis"),
            ("tiktok", "short_video", "Heartbreaking post-match press conference from Carlos Alcaraz: 'Mentally, I took steps back. I didn't know how to control myself on court today.' #alcaraz #usopen #tennis #reaction"),
            ("tiktok", "short_video", "Novak Djokovic serves a career-worst 14 double faults as Alexei Popyrin eliminates the 24-time Grand Slam champion! End of an era? #djokovic #tennis #usopen"),
            ("youtube", "official_channel", "MASSIVE SHOCK! Botic van de Zandschulp Stuns Carlos Alcaraz in Straight Sets | US Open Highlights"),
            ("youtube", "channel", "Why Did Carlos Alcaraz and Novak Djokovic Both Suffer Early Exits at the US Open? Olympic Hangover Analysis"),
            ("reddit", "subreddit", "[Post-Match Thread] Botic van de Zandschulp defeats [3] Carlos Alcaraz 6-1, 7-5, 6-4 (US Open R2)"),
            ("reddit", "subreddit", "[Post-Match Thread] [28] Alexei Popyrin defeats [2] Novak Djokovic 6-4, 6-4, 2-6, 6-4 (US Open R3)"),
            ("news", "sports_news", "US Open stunned as Carlos Alcaraz crashes out in straight-sets second-round shock against Van de Zandschulp"),
            ("news", "sports_news", "Novak Djokovic follows Carlos Alcaraz out of US Open in shock third-round defeat to Alexei Popyrin")
        ],
        "variations": [
            "The physical and emotional toll of the Paris Olympic final clearly caught up with both finalists in New York.",
            "Botic van de Zandschulp had considered retiring from tennis just three months ago and now produces the performance of his life.",
            "The men's draw at Flushing Meadows is blown wide open for Jannik Sinner, Daniil Medvedev, and Alexander Zverev.",
            "Djokovic finishes a calendar year without winning a Grand Slam title for the first time since 2017.",
            "Fans on Arthur Ashe Stadium witnessed two of the most stunning upsets on back-to-back night sessions."
        ]
    },
    {
        "id": "usopen_finals_sinner_sabalenka_champions",
        "name": "US Open Finals: Sinner & Sabalenka Crowned Champions",
        "sport": "tennis",
        "peak_day": 23, # Sep 8
        "duration": 5,
        "is_drama": False,
        "base_sentiment": 0.65,
        "controversy": 0.4,
        "hashtags": ["usopen", "sinner", "sabalenka", "tennis", "champion", "grandslam"],
        "templates": [
            ("facebook", "fanpage", "US OPEN CHAMPION: World No. 1 Jannik Sinner sweeps past home hope Taylor Fritz in straight sets (6-3, 6-4, 7-5) to capture his second Grand Slam title of 2026! An incredible triumph amidst intense scrutiny! #sinner #usopen #tennis"),
            ("facebook", "fanpage", "Aryna Sabalenka powers past Jessica Pegula 7-5, 7-5 to win her maiden US Open title! After falling short in last year's final, the World No. 2 lifts her third Grand Slam trophy in dominant fashion! #sabalenka #usopen #tennis"),
            ("tiktok", "short_video", "Emotional moment Jannik Sinner lifts the US Open trophy and dedicates the victory to his aunt who is dealing with health issues #sinner #usopen #tennis #emotional"),
            ("tiktok", "short_video", "Aryna Sabalenka dancing and celebrating with her team after sealing championship point on Arthur Ashe Stadium! Pure joy! #sabalenka #usopen #tennis"),
            ("youtube", "official_channel", "JANNIK SINNER VS TAYLOR FRITZ: US OPEN 2026 MEN'S FINAL EXTENDED HIGHLIGHTS"),
            ("youtube", "official_channel", "ARYNA SABALENKA CROWNED US OPEN CHAMPION | Trophy Presentation & Winning Speech"),
            ("reddit", "subreddit", "[Post-Match Thread] [1] Jannik Sinner defeats [12] Taylor Fritz 6-3, 6-4, 7-5 to win the 2026 US Open!"),
            ("reddit", "subreddit", "[Post-Match Thread] [2] Aryna Sabalenka defeats [6] Jessica Pegula 7-5, 7-5 to claim her maiden US Open title"),
            ("news", "sports_news", "Jannik Sinner sweeps past Taylor Fritz to claim historic US Open title and second Grand Slam crown of 2026"),
            ("news", "sports_news", "Aryna Sabalenka powers past home favorite Jessica Pegula to win first US Open crown in New York")
        ],
        "variations": [
            "Sinner becomes the first Italian man in tennis history to capture the US Open singles championship.",
            "Taylor Fritz fought bravely in the third set, but Sinner's baseline consistency proved unbreachable.",
            "Sabalenka showed immense mental resilience to recover from 3-5 down in the second set to close out the final.",
            "American tennis fans will have to wait longer to end their Grand Slam men's singles drought dating back to Andy Roddick in 2003.",
            "With two hard-court Slams in 2026, Jannik Sinner firmly cements his position atop the ATP rankings."
        ]
    },
    {
        "id": "north_london_derby_spurs_arsenal",
        "name": "North London Derby: Tottenham vs Arsenal Referee Clash",
        "sport": "football",
        "peak_day": 29, # Sep 14
        "duration": 4,
        "is_drama": True,
        "base_sentiment": 0.25,
        "controversy": 0.72,
        "hashtags": ["arsenal", "tottenham", "northlondonderby", "gabriel", "arteta", "football"],
        "templates": [
            ("facebook", "fanpage", "NORTH LONDON IS RED: Despite missing Martin Odegaard and Declan Rice, Arsenal defeat Tottenham 1-0 away from home thanks to Gabriel's bullet header! What a defensive masterclass by Mikel Arteta! #arsenal #tottenham #derby #premierleague"),
            ("facebook", "fanpage", "Heated controversy in the North London Derby: Tottenham fans and pundits argue Gabriel pushed Cristian Romero before scoring. VAR Jarred Gillett confirmed the goal. Was it a clear foul? Let us know your thoughts! #tottenham #arsenal #var #premierleague"),
            ("tiktok", "short_video", "Gabriel Magalhaes rising highest to power past Vicario and silence the Tottenham Hotspur Stadium! Huge derby moment! #gabriel #arsenal #tottenham #derby"),
            ("tiktok", "short_video", "Heated brawl between Jurrien Timber and Guglielmo Vicario in the first half: referee dishing out yellow cards left and right! #arsenal #spurs #brawl #premierleague"),
            ("youtube", "official_channel", "TOTTENHAM 0-1 ARSENAL | GABRIEL BULLET HEADER DECIDES FIERY NORTH LONDON DERBY | EXTENDED HIGHLIGHTS"),
            ("youtube", "channel", "Tactical Masterclass: How Mikel Arteta Set Up Arsenal's Low Block to Completely Neutralize Ange Postecoglou"),
            ("reddit", "subreddit", "[Post-Match Thread] Tottenham Hotspur 0-1 Arsenal (North London Derby)"),
            ("reddit", "subreddit", "VAR Check on Gabriel challenge on Romero prior to goal: Was there an illegal push? Sky Sports debate clip"),
            ("news", "sports_news", "Arsenal overcome absences of Declan Rice and Martin Odegaard to beat Tottenham in fiery North London Derby"),
            ("news", "sports_news", "Ange Postecoglou defends Tottenham set-piece defensive record despite costly derby defeat to Arsenal")
        ],
        "variations": [
            "Arsenal record their third consecutive away victory at the Tottenham Hotspur Stadium for the first time since the 1980s.",
            "William Saliba and Gabriel produced a heroic central defensive partnership to deny Spurs any clear-cut openings.",
            "Ange Postecoglou faces mounting scrutiny over Tottenham's vulnerability defending dead-ball situations.",
            "Jurrien Timber was fortunate to escape a red card following a contentious confrontation near the touchline.",
            "Mikel Arteta celebrates a massive psychological boost ahead of Arsenal's Champions League opener and clash with Man City."
        ]
    },
    {
        "id": "ucl_new_swiss_format_player_strike",
        "name": "UEFA Champions League 36-Team Swiss Format & Player Strike",
        "sport": "football",
        "peak_day": 28, # Sep 13
        "duration": 5,
        "is_drama": True,
        "base_sentiment": -0.2,
        "controversy": 0.68,
        "hashtags": ["ucl", "championsleague", "rodri", "uefa", "playerstrike", "football"],
        "templates": [
            ("facebook", "fanpage", "MAJOR CONTROVERSY: Manchester City star Rodri and Liverpool keeper Alisson warn players are close to striking over the grueling football schedule and the new 36-team Champions League format: 'We are not robots. If this continues, we have no other option.' Do you support the players? #rodri #ucl #championsleague #football"),
            ("facebook", "fanpage", "The new UEFA Champions League format begins this week: 36 teams in one massive league table, no traditional groups, 8 matches instead of 6. Will this revitalise the tournament or cause massive player burnout? #championsleague #uefa #football"),
            ("tiktok", "short_video", "Rodri speaking the brutal truth about the modern football calendar: 'Someone has to take care of us. 70 games a season is unsustainable.' #rodri #mancity #ucl #football"),
            ("tiktok", "short_video", "The new 36-team UEFA Champions League 'Swiss Model' format explained in 60 seconds so you won't get confused! #ucl #championsleague #football #fyp"),
            ("youtube", "official_channel", "UEFA CHAMPIONS LEAGUE NEW FORMAT EXPLAINED: Complete Guide to the 36-Team Swiss Model"),
            ("youtube", "channel", "Are Footballers on the Verge of a Historic Strike? Investigating UEFA and FIFA's Expanding Schedules"),
            ("reddit", "subreddit", "Rodri on potential player strike over fixture congestion: 'I think we are close to that. If it keeps this way, we have no other option.'"),
            ("reddit", "subreddit", "The 36-Team UEFA Champions League 'Swiss Model' begins this week: Discussion & Predictions thread"),
            ("news", "sports_news", "Rodri warns players could strike over schedule as Champions League expands to 36 teams"),
            ("news", "sports_news", "UEFA Champions League: How the new Swiss-model league phase works and why it divides fans and managers")
        ],
        "variations": [
            "With the expanded Champions League and the upcoming 32-team FIFA Club World Cup, elite players face up to 75 games in a single year.",
            "Managers across Europe including Pep Guardiola and Carlo Ancelotti backed Rodri's comments regarding player welfare.",
            "Broadcasters and UEFA defend the new format, promising more blockbuster matches between top European clubs in the opening phase.",
            "FIFPRO, the international players' union, is taking legal action against FIFA over the unilateral imposition of international fixtures.",
            "Fans express mixed emotions, torn between excitement for top matches and sympathy for overworked athletes."
        ]
    },
    {
        "id": "monza_f1_leclerc_ferrari_miracle",
        "name": "Formula 1 Italian GP Monza: Leclerc & Ferrari Miracle Win",
        "sport": "motorsport",
        "peak_day": 16, # Sep 1
        "duration": 4,
        "is_drama": False,
        "base_sentiment": 0.75,
        "controversy": 0.25,
        "hashtags": ["f1", "leclerc", "ferrari", "monza", "italiangp", "mclaren"],
        "templates": [
            ("facebook", "fanpage", "MIRACLE AT MONZA: Charles Leclerc and Scuderia Ferrari execute a bold one-stop strategy to hold off McLaren and win the Italian Grand Prix in front of hundreds of thousands of ecstatic Tifosi! Pure motorsport magic! #f1 #ferrari #leclerc #monza"),
            ("facebook", "fanpage", "McLaren's 'Papaya Rules' debate explodes: Oscar Piastri's aggressive Lap 1 overtake on Lando Norris opened the door for Ferrari's win. Did McLaren throw away a guaranteed 1-2 finish? #f1 #mclaren #norris #piastri"),
            ("tiktok", "short_video", "The deafening roar of the Tifosi at Monza as Charles Leclerc crosses the finish line to take victory for Ferrari! Goosebumps guaranteed! #f1 #leclerc #ferrari #monza #fyp"),
            ("tiktok", "short_video", "Emotional team radio of Charles Leclerc shouting in Italian after claiming his second Monza victory! What a moment! #f1 #charlesleclerc #ferrari"),
            ("youtube", "official_channel", "RACE HIGHLIGHTS: 2026 ITALIAN GRAND PRIX | CHARLES LECLERC WINS ON ONE-STOP THRILLER AT MONZA"),
            ("youtube", "channel", "How Ferrari and Charles Leclerc Pulled Off the Greatest Strategic Masterclass in Modern Monza History"),
            ("reddit", "subreddit", "[Post-Race Thread] 2026 Italian Grand Prix - Monza (Charles Leclerc wins for Ferrari!)"),
            ("reddit", "subreddit", "McLaren's 'Papaya Rules' debated as Oscar Piastri's Lap 1 overtake compromises Lando Norris's title bid against Max Verstappen"),
            ("news", "sports_news", "Charles Leclerc wins Italian Grand Prix for Ferrari with inspired one-stop strategy at Monza"),
            ("news", "sports_news", "McLaren admit they will review team orders after Monza defeat to Ferrari damages Norris title chase")
        ],
        "variations": [
            "Tyre management from Charles Leclerc in the final 15 laps was a masterclass under immense pressure from Piastri.",
            "Max Verstappen could only manage 6th place as Red Bull's handling balance issues continued at high-speed Monza.",
            "The iconic podium celebration over the Monza main straight was painted in an ocean of red flags and flares.",
            "McLaren boss Andrea Stella admitted post-race that the team will consider prioritizing Norris in future races.",
            "Ferrari's upgraded floor and aerodynamic package proved crucial in reducing rear tyre degradation throughout the stint."
        ]
    },
    {
        "id": "real_madrid_mbappe_vinicius_tactical_debate",
        "name": "Real Madrid Mbappe & Vinicius Chemistry Debate",
        "sport": "football",
        "peak_day": 8, # Aug 24
        "duration": 8,
        "is_drama": True,
        "base_sentiment": 0.1,
        "controversy": 0.6,
        "hashtags": ["realmadrid", "mbappe", "vinicius", "laliga", "ancelotti", "halamadrid"],
        "templates": [
            ("facebook", "fanpage", "Carlo Ancelotti's tactical puzzle: Kylian Mbappe and Vinicius Junior occupying the same left-wing spaces as Real Madrid drop early points in La Liga draw at Mallorca! Can the stars coexist? #realmadrid #mbappe #vinicius #laliga"),
            ("facebook", "fanpage", "Relief at the Santiago Bernabeu: Kylian Mbappe scores his first two La Liga goals in Real Madrid's 2-0 win over Real Betis, assisted by a sublime backheel from Fede Valverde! The Galacticos are waking up! #mbappe #realmadrid #laliga"),
            ("tiktok", "short_video", "Kylian Mbappe celebrating his maiden La Liga goal at the Bernabeu with his iconic arms-crossed celebration in front of 80,000 Madridistas #mbappe #realmadrid #laliga #fyp"),
            ("tiktok", "short_video", "Vinicius Junior unselfishly handing the penalty to Kylian Mbappe to silence the media critics and kill the rift rumors #vinicius #mbappe #realmadrid"),
            ("youtube", "official_channel", "REAL MADRID 2-0 REAL BETIS | KYLIAN MBAPPE SCORES FIRST LA LIGA BRACE AT SANTIAGO BERNABEU | EXTENDED HIGHLIGHTS"),
            ("youtube", "channel", "Tactical Analysis: How Carlo Ancelotti Solved Real Madrid's Attacking Spacing Between Mbappe and Vinicius"),
            ("reddit", "subreddit", "[Post-Match Thread] Real Madrid 2-0 Real Betis (Kylian Mbappe brace breaks league duck)"),
            ("reddit", "subreddit", "Tactical Discussion: How Ancelotti is adjusting Real Madrid's forward pressing without Toni Kroos directing tempo"),
            ("news", "sports_news", "Kylian Mbappe scores twice as Real Madrid defeat Real Betis to ease early La Liga anxieties"),
            ("news", "sports_news", "Carlo Ancelotti praises Vinicius Junior selflessness in setting up Mbappe for Real Madrid breakthrough")
        ],
        "variations": [
            "Mbappe admitted in his post-match interview that he felt immense relief after going three league games without a goal.",
            "Spanish media had mounted intense scrutiny on the French superstar before his clinical second-half double.",
            "Fede Valverde continues to establish himself as one of the most complete box-to-box midfielders in world football.",
            "Real Madrid remain four points behind Barcelona after the opening four rounds of the Spanish top flight.",
            "The synergy between Rodrygo, Vinicius, and Mbappe showed significant improvement compared to the opening fixture."
        ]
    },
    {
        "id": "barcelona_flick_revolution_yamal",
        "name": "Barcelona Hansi Flick Revolution & Lamine Yamal Brilliance",
        "sport": "football",
        "peak_day": 12, # Aug 28
        "duration": 6,
        "is_drama": False,
        "base_sentiment": 0.8,
        "controversy": 0.3,
        "hashtags": ["barca", "barcelona", "lamineyamal", "hansiflick", "laliga", "pedri"],
        "templates": [
            ("facebook", "fanpage", "THE HANSI FLICK REVOLUTION: FC Barcelona start the La Liga season with four consecutive victories, scoring 13 goals and displaying breathtaking high-intensity football! The German tactician has transformed this side! #barcelona #hansiflick #laliga"),
            ("facebook", "fanpage", "Lamine Yamal is playing like a seasoned veteran at just 17 years old: 4 games, 1 goal, 4 assists! The Spanish prodigy continues to dazzle European football with mesmerizing skill! #lamineyamal #barca #laliga"),
            ("tiktok", "short_video", "Lamine Yamal leaving defenders completely twisted with his signature body feints and trivela crosses in La Liga! Absolutely generational talent! #lamineyamal #barca #skills #football"),
            ("tiktok", "short_video", "Barcelona's aggressive high-press under Hansi Flick is suffocating opposition teams: fitness levels are night and day compared to last season! #barcelona #hansiflick #football"),
            ("youtube", "official_channel", "BARCELONA 7-0 REAL VALLADOLID | RAPHINHA HAT-TRICK & LAMINE YAMAL MASTERCLASS | EXTENDED HIGHLIGHTS"),
            ("youtube", "channel", "How Hansi Flick Transformed Barcelona into a Relentless Pressing Machine in Just Two Months"),
            ("reddit", "subreddit", "[Post-Match Thread] FC Barcelona 7-0 Real Valladolid (La Liga Matchweek 4)"),
            ("reddit", "subreddit", "Lamine Yamal stats at 17 years old: Most progressive carries and expected assists in Europe's top 5 leagues this season"),
            ("news", "sports_news", "Hansi Flick's energetic Barcelona maintain perfect start with scintillating 7-0 rout of Valladolid"),
            ("news", "sports_news", "Lamine Yamal continues meteoric rise as Barcelona find joy and intensity under new German manager")
        ],
        "variations": [
            "Raphinha scored his first career hat-trick in Barcelona colours, thriving in an inverted central role.",
            "Robert Lewandowski looks revitalised leading the line under his former Bayern Munich manager.",
            "Despite ongoing financial fair play restrictions, Barcelona are setting the pace at the summit of La Liga.",
            "Pau Cubarsi and Inigo Martinez have formed a remarkably disciplined offside trap in the defensive line.",
            "Yamal's vision and decision-making in the final third draw natural comparisons to young Lionel Messi."
        ]
    },
    {
        "id": "nba_olympics_aftermath_curry_lebron",
        "name": "Olympic Basketball Legacy: Curry & LeBron Team USA Dominance",
        "sport": "basketball",
        "peak_day": 4, # Aug 19-20
        "duration": 6,
        "is_drama": False,
        "base_sentiment": 0.85,
        "controversy": 0.2,
        "hashtags": ["nba", "curry", "lebron", "teamusa", "basketball", "olympics"],
        "templates": [
            ("facebook", "fanpage", "Stephen Curry's historic 3-point barrage to secure Olympic Gold for Team USA in Paris is still dominating global basketball debates! Truly the greatest shooter in the history of the sport! #curry #nba #teamusa #basketball"),
            ("facebook", "fanpage", "LeBron James and Stephen Curry sharing the court for Team USA: Will we ever see an NBA blockbuster trade uniting the two legends before they retire? Speculation heats up! #lebron #curry #lakers #warriors #nba"),
            ("tiktok", "short_video", "Stephen Curry hitting the iconic 'Nuit Nuit' (Night Night) contested 3-pointer over three French defenders to seal the Olympic Gold medal! Unforgettable! #curry #nightnight #olympics #nba #fyp"),
            ("tiktok", "short_video", "LeBron James' reaction to Steph Curry hitting four straight three-pointers in the final two minutes: complete disbelief and admiration! #lebron #curry #teamusa #nba"),
            ("youtube", "official_channel", "STEPHEN CURRY 4 CLUTCH 3S IN 2 MINUTES TO WIN GOLD! Full Uncut Sequence | Team USA vs France"),
            ("youtube", "channel", "Could LeBron James and Stephen Curry Actually Team Up in the NBA Next Season? Trade Scenarios Analyzed"),
            ("reddit", "subreddit", "Looking back at Stephen Curry's historic 4th quarter in the Olympic Gold Medal game: The greatest clutch shooting sequence ever?"),
            ("reddit", "subreddit", "Steve Kerr on coaching LeBron James and Steph Curry together: 'A once-in-a-lifetime privilege.'"),
            ("news", "sports_news", "Stephen Curry's golden flurry in Paris cements legacy alongside LeBron James in Team USA folklore"),
            ("news", "sports_news", "NBA stars return to training camps energized after historic Olympic tournament in Paris")
        ],
        "variations": [
            "Curry scored 36 points in the semifinal comeback against Serbia followed by 24 points in the final against France.",
            "LeBron James was awarded the tournament MVP after averaging a near triple-double throughout the competition.",
            "Victor Wembanyama's emotional tears after the final showed just how much pressure the French team carried on home soil.",
            "The mutual respect between the two generational superstars defined the final chapter of their international careers.",
            "Fans around the world consider the 2024 Team USA roster the closest spiritual successor to the 1992 Dream Team."
        ]
    }
]

def generate_english_sports_dataset(target_total_rows: int = 2150):
    logger.info(f"Generating 100% ENGLISH 30-day Multi-Platform Sports Social Listening Dataset ({target_total_rows} rows)...")
    logger.info(f"Temporal Window: {START_DATE.strftime('%Y-%m-%d')} to {(START_DATE + timedelta(days=DAYS_COUNT)).strftime('%Y-%m-%d')}")

    records = []
    post_counter = 1

    platform_metric_ranges = {
        "facebook": {"likes": (40, 2500), "shares": (5, 450), "comments": (10, 800), "views": (500, 15000), "followers": (10000, 800000)},
        "tiktok": {"likes": (150, 18000), "shares": (20, 2200), "comments": (25, 1500), "views": (2000, 250000), "followers": (5000, 500000)},
        "youtube": {"likes": (80, 5000), "shares": (10, 300), "comments": (15, 600), "views": (1500, 80000), "followers": (20000, 1500000)},
        "reddit": {"likes": (15, 650), "shares": (1, 35), "comments": (8, 250), "views": (0, 0), "followers": (100, 2000)},
        "news": {"likes": (10, 300), "shares": (5, 80), "comments": (2, 90), "views": (500, 12000), "followers": (50000, 2000000)},
    }

    english_user_prefixes = [
        "", "", "", "Breaking update: ", "Personal thoughts on this: ", "Hot topic today: ",
        "Still can't wrap my head around this: ", "Huge talking point: ", "Full story: ",
        "Official confirmation: ", "Unbelievable scenes: ", "My honest opinion: ",
        "Big controversy brewing: ", "Just in: ", "Here is the reality: "
    ]

    posts_per_arc = target_total_rows // len(TOPIC_ARCS)
    remainder = target_total_rows % len(TOPIC_ARCS)

    for idx, arc in enumerate(TOPIC_ARCS):
        num_posts_this_arc = posts_per_arc + (1 if idx < remainder else 0)
        peak_day = arc["peak_day"]
        duration = arc["duration"]
        templates = arc["templates"]
        variations = arc["variations"]

        for _ in range(num_posts_this_arc):
            day_offset = int(round(random.gauss(peak_day, duration / 2.3)))
            day_offset = max(0, min(DAYS_COUNT - 1, day_offset))

            hour_weights = [1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 9, 8, 7, 7, 8, 9, 10, 12, 14, 15, 14, 10, 5]
            hour = random.choices(range(24), weights=hour_weights)[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)

            post_time = START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
            time_str = post_time.strftime("%Y-%m-%d %H:%M:%S")

            tpl_platform, tpl_source, tpl_text = random.choice(templates)
            prefix = random.choice(english_user_prefixes)
            variation_text = random.choice(variations) if random.random() < 0.65 else ""

            if variation_text:
                full_text = f"{prefix}{tpl_text} {variation_text}".strip()
            else:
                full_text = f"{prefix}{tpl_text}".strip()

            m_ranges = platform_metric_ranges[tpl_platform]
            is_near_peak = abs(day_offset - peak_day) <= 1
            mult = random.uniform(1.8, 3.5) if is_near_peak else random.uniform(0.6, 1.2)

            likes = int(random.randint(*m_ranges["likes"]) * mult)
            shares = int(random.randint(*m_ranges["shares"]) * mult)
            comments = int(random.randint(*m_ranges["comments"]) * mult)
            views = int(random.randint(*m_ranges["views"]) * mult) if m_ranges["views"][1] > 0 else 0
            followers = random.randint(*m_ranges["followers"])

            if arc["is_drama"]:
                comments = int(comments * random.uniform(1.4, 2.2))

            hashtags = list(set(arc["hashtags"] + [tpl_platform, arc["sport"]]))

            prefix_id = {"facebook": "fb", "tiktok": "tt", "youtube": "yt", "reddit": "rd", "news": "news"}[tpl_platform]
            post_id = f"{prefix_id}_{post_time.strftime('%Y%m%d')}_{post_counter:05d}"
            post_counter += 1

            records.append({
                "post_id": post_id,
                "text": full_text,
                "timestamp": time_str,
                "likes": likes,
                "shares": shares,
                "comments": comments,
                "views": views,
                "hashtags": hashtags,
                "followers": followers,
                "platform": tpl_platform,
                "sport_category": arc["sport"],
                "source_type": tpl_source
            })

    df = pd.DataFrame(records)
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    logger.info(f"English dataset generation complete: {len(df)} records across {df['platform'].nunique()} platforms.")
    logger.info(f"Platform distribution:\n{df['platform'].value_counts().to_string()}")
    logger.info(f"Sport category distribution:\n{df['sport_category'].value_counts().to_string()}")
    logger.info(f"Time range: from {df['timestamp'].min()} to {df['timestamp'].max()} ({DAYS_COUNT} days)")

    csv_path = os.path.join(OUTPUT_DIR, "social_listening_30days.csv")
    parquet_path = os.path.join(OUTPUT_DIR, "raw_posts.parquet")
    input_parquet_path = os.path.join(OUTPUT_DIR, "input_data.parquet")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_parquet(parquet_path, index=False)
    df.to_parquet(input_parquet_path, index=False)

    logger.info(f"Saved: {csv_path}")
    logger.info(f"Saved: {parquet_path}")
    logger.info(f"Saved: {input_parquet_path}")
    return df

if __name__ == "__main__":
    generate_english_sports_dataset(2150)
