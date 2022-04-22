# 反转链表系列

## Leetcode206. 反转链表

### 迭代：

```java
public ListNode reverseList(ListNode head) {
    if (head == null)
        return null;
    if (head.next == null) 
        return head;
    
    ListNode cur = head;
    ListNode p = cur.next;
    ListNode helper = p.next;
    if (helper == null) {
        cur.next = null;
        p.next = cur;
        return p;
    }
    cur.next = null;
    
    while (true) {
        if (p == null)
            break;
        p.next = cur;
        cur = p;
        p = helper;
        if (helper != null)
            helper = helper.next;
    }
    
    return cur;
}
```

### 递归：

```java
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null)
        return head;
    //这里反转链表不会改变前面链表的next指针，因此 head 的next其实还指着原来链表结构中的它的指向节点。
    ListNode last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
}
```

## 反转链表前N个节点

代码：

```java
ListNode succsussor = null;

ListNode reverseN(ListNode head, int n) {
    if (n == 1) {
        succussor = head.next;
        return head;
    }
    ListNode last = reverseN(head.next, n - 1);
    head.next.next = head;
    head.next = succsussor;
    return last;
}
```



## 反转链表的一部分[n, m]

```java
static ListNode reverseBetween(ListNode head, int m, int n) {
    if (m == 1) {
        //如果 m = 1相当于反转前 n 个链表
        return reverseN(head, n);
    }
    head.next = reverseBetween(head.next, m - 1, n - 1);
    return head;
}
```



## Leetcode25. k 个一组反转链表

将链表按 k 个反转

思路：

1. 先反转以 head 开头的 k 个元素
2. 将第 k+ 1 个元素作为 head 递归调用 `reverseKGroup` 函数
3. 将上述两个过程的结果连接起来

代码：

首先实现迭代反转链表：

```java
ListNode reverse(ListNode head) {
    ListNode pre = null, cur = head, nxt = head;
    while (cur != null) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    return pre;
}
```

如果反转 a 到 b 的链表：

```java
ListNode reverse(ListNode a, ListNode b) {
    ListNode pre, cur, nxt;
    pre = null, cur = a, nxt = a;
    while (cur != b) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    
    return pre;
}
```

最终代码：

```java
private ListNode reverse(ListNode a, ListNode b) {
    ListNode pre, cur, nxt;
    pre = null; cur = a; nxt = a;
    while (cur != b) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur; 
        cur = nxt;
    }
    return pre;
}
public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    ListNode a = head; ListNode b = head;
    for (int i = 0; i < k; i++) {
        if (b == null) return head;
        b = b.next;
    }
    
    ListNode newHead = reverse(a, b);  //reverse 一段后该段的头结点
    a.next = reverseKGroup(b, k);  //递归连接两段
    
    return newHead;
}
```



# Leetcode146. LRU 缓存

本题使用 哈希表 + 双端链表 解决。

哈希表存放所有的节点，key 是 存放的key，value是对应key的节点。

双端链表的头结点的后面一个节点存放最近最不经常使用的节点，双端链表的尾结点的前一个节点是最近使用的节点。

代码：

````java
class LRUCache {

    private final Map<Integer, ListNode> map;
    private final ListNode head;
    private final ListNode tail;
    private final int capacity;
    
    public LRUCache(int capacity) {
        map = new HashMap<>(capacity);
        head = new ListNode();
        tail = new ListNode();
        head.next = tail;
        tail.prev = head;
        this.capacity = capacity;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        moveToTail(key);
        return map.get(key).val;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            map.get(key).val = value;
            moveToTail(key);
            return;
        }
        if (map.size() >= capacity) {
            int k = head.next.key;
            map.remove(k);
            head.next = head.next.next;
            head.next.prev = head;
        }
        ListNode node = new ListNode();
        node.key = key; node.val = value;
        map.put(key, node);
        
        node.next = tail;
        node.prev = tail.prev;
        
        tail.prev.next = node;
        tail.prev = node;
    }

    private void moveToTail(int key){
        ListNode node = map.get(key);
        
        node.prev.next = node.next;
        node.next.prev = node.prev;
        
        node.next = tail;
        node.prev = tail.prev;
        
        tail.prev.next = node;
        tail.prev = node;
    }
    
    private static class ListNode {
        ListNode prev;
        ListNode next;
        int key;
        int val;
        ListNode () {}
    }
    
}
````

# Leetcode3.  无重复字符的最长子串

本题使用滑动窗口解决：

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> map = new HashMap<>();
        int l = 0, r = 0;
        int res = 0;
        while (r < s.length()) {
            Character c = s.charAt(r);
            r ++;
            map.put(c, map.getOrDefault(c, 0) + 1);

            while (map.getOrDefault(c, 0) > 1) {
                Character d = s.charAt(l);
                map.put(d, map.getOrDefault(d, 0) - 1);
                l ++;
            }

            res = Math.max(res, r - l);
        }
        return res;
    }
}
```

# Leetcode912. 排序数组

将数组升序排序

## 归并排序

代码：

```java
int[] temp;

int[] sort(int[] nums) {
    if (nums == null || nums.length <= 1) return nums; 
    temp = new int[temp.length];
    mergeSort(nums, 0, nums.length - 1);
    return nums;
}

private mergeSort(int[] nums, int l, int r) {
    if (l >= r) return;
    int mid = l + r >> 1;
    mergeSort(nums, l, mid);
    mergeSort(nums, mid + 1, r);
    int i = l, j = mid + 1, k = 0
    while (i <= mid && j <= r) {
        if (nums[i] < nums[j]) 
            temp[k ++] = nums[i ++];
        else 
            temp[k ++] = nums[j ++];
    }
    
    while (i <= mid)
        temp[k ++] = nums[i ++];
    while (j <= r)
        temp[k ++] = nums[j ++];
	
    for (int index = l, helper = 0; index <= r; index ++, helper ++) 
        nums[index] = temp[helper];
}
```



## 快速排序

代码：

```java
int[] sort(int[] nums) {
    if (nums == null || nums.length <= 1) return nums; 
    quickSort(nums, 0, nums.length - 1);
    return nums;
}

void quickSort(int[] nums, int l, int r) {
    if (l >= r) return;
    int start = l - 1, end = r + 1;
    int x = nums[l + r >> 1];
    
    while (start < end) {
        do { start ++; } while (nums[start] < x);
        do { end --; } while (nums[end] > x);
    	if ( start < end ) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
        }
    }
    
    quickSort(nums, l, end);
    quickSort(nums, end + 1, r);
}
```

## 堆排序

```java
public int[] sortArray(int[] nums) {
    if (nums == null || nums.length <= 1) return nums;
    
    int n = nums.length;
    
    for (int i = n / 2 - 1; i >= 0; i--) {
        adjustHeap(nums, i, n);
    }
    
    for (int i = n - 1; i >= 0; i--) {
        int temp = nums[i];
        nums[i] = nums[0];
        nums[0] = temp;
        adjustHeap(nums, 0, i);
    }
    
    return nums;
}

// CORE!!!
private void adjustHeap(int[] nums, int i, int length) {
    int temp = nums[i];
    for (int j = 2 * i + 1; j < length; j = j * 2 + 1) {
        if (j + 1 < length && nums[j + 1] > nums[j]) {
            j ++;
        }
        if (temp < nums[j]) {
            nums[i] = nums[j];
            i = j;
        } else {
            break;
        }
    }
    nums[i] = temp;
}
```

# Leetcode15. 三数之和

for 循环一个端点，另外两个端点通过双指针，一个指针指向第一个元素，另外一个指针指向最后一个元素，通过指针滑动来寻找。

代码：

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Set<List<Integer>> set = new HashSet<>();
        if (nums == null) return null;
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i <= n - 3; i++) {
            int l = i + 1, r = n - 1;
            while (l < r) {
                if (nums[i] + nums[l] + nums[r] < 0)
                    l ++;
                else if (nums[i] + nums[l] + nums[r] > 0)
                    r --;
                else {
                    List<Integer> temp = new ArrayList<>();
                    temp.add(nums[i]); temp.add(nums[l]); temp.add(nums[r]);
                    if (!set.contains(temp)){
                        res.add(temp);
                        set.add(temp);
                    }
                    l ++;
                    r --;
                }
            }
        }
        return res;
    }
}
```

这里我为了判重所以使用了一个 HashSet，但是判重还可以不用Set来降低空间复杂度。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null) return null;
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i <= n - 3; i++) {
            // 判重，如果重复则 continue
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            
            int l = i + 1, r = n - 1;
            while (l < r) {
                if (nums[i] + nums[l] + nums[r] < 0)
                    l ++;
                else if (nums[i] + nums[l] + nums[r] > 0)
                    r --;
                else {
                    List<Integer> temp = new ArrayList<>();
                    temp.add(nums[i]); temp.add(nums[l]); temp.add(nums[r]);
                    //判重，如果重复则移动指针，但是要注意数组范围不要越界
                    while (l < r && nums[l] == nums[l + 1])
                        l ++;
                    while (l < r && nums[r] == nums[r - 1])
                        r --;
                    l ++;  //移动指针
                    r --;  //移动指针
                    
                    res.add(temp);
                }
            }
        }
        return res;
    }
}
```

# Leetcode53. 最大子数组和

这题其实可以用动态规划：

`dp[i]`：表示以索引 i 结尾的最大子数组和。

状态是索引的位置，选择是选择当前的数还是不选当前索引的数。

状态转移方程：

```
dp[i] = dp[i - 1] + nums[i] (dp[i - 1] > 0)
		nums[i]             (dp[i - 1] < 0)
```

代码：

```java
public int maxSubArray(int[] nums) {
    if (nums == null || nums.length <= 0) return 0;
    int[] dp = new int[nums.length];
    System.arraycopy(nums, 0, dp, 0, nums.length);
    
    for (int i = 1; i < nums.length; i++) {
        if (dp[i - 1] < 0)
            dp[i] = nums[i];
        else
            dp[i] = nums[i] + dp[i - 1];
    }
    
    int res = Integer.MIN_VALUE;
    
    for (int i : dp) {
        res = Math.max(res, i);
    }
    return res;
}
```

状态压缩：

```java
public int maxSubArray(int[] nums) {
    if (nums == null || nums.length <= 0) return 0;
    
    int temp = nums[0];
    int res = temp;
    
    for (int i = 1; i < nums.length; i++) {
        if (temp > 0)
            temp = temp + nums[i];
        else
            temp = nums[i];
    
        res = Math.max(res, temp);
    }
    
    return res;
}
```



# Leetcode21. 合并两个有序链表

这里要注意使用虚拟头结点，代码会好些很多。

```java
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    if (list1 == null) 
        return list2;
    if (list2 == null)
        return list1;
    // 虚拟头结点
    ListNode preHead = new ListNode(-1);
    ListNode pre = preHead;
    
    ListNode l1 = list1, l2 = list2;    
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            pre.next = l1;
            l1 = l1.next;
        } else {
            pre.next = l2;
            l2 = l2.next;
        }
        pre = pre.next;
    }
    
    if (l1 != null)
        pre.next = l1;
    if (l2 != null)
        pre.next = l2;
    return preHead.next;
}
```

递归写法：

```java
// 首先明确mergeTwoLists 方法的意义：合并传入的两条链表并返回第一个节点
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    if (list1 == null) {
        return list2;
    } else if (list2 == null) {
        return list1;
    } else if (list1.val < list2.val) {
        list1.next = mergeTwoLists(list1.next, list2);
        return list1;
    } else {
        list2.next = mergeTwoLists(list1, list2.next);
        return list2;
    }
}
```



# Leetcode1. 两数之和

我们用一个 `HashMap` 记录之前出现过的数，key 是target - nums[j]，value 是 j（对应下标）

```java
public int[] twoSum(int[] nums, int target) {
    if (nums == null || nums.length < 2)
        return null;
    int[] res = new int[2];
    Map<Integer, Integer> map = new HashMap<>();
    
    for (int i = 0; i < nums.length; i++) {
        if (map.containsKey(nums[i])) {
            res[0] = map.get(nums[i]);
            res[1] = i;
            return res;
        }
        
        map.put(target - nums[i], i);
    }
    return null;
}
```



# Leetcode141. 环形链表

使用快慢指针，fast一次走两步，slow一次走一步。

```java
public boolean hasCycle(ListNode head) {
    if (head == null) return false;
    ListNode fast = head, slow = head;
    while (fast != null) {
        fast = fast.next;
        slow = slow.next;
        
        if (fast == null)
            break;
        fast = fast.next;
        if (fast == slow)
            return true;
    }
    return false;
}
```



# Leetcode102. 二叉树的层序遍历

使用 BFS 算法

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root == null) return res;
    
    Queue<TreeNode> queue = new ArrayDeque<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        List<Integer> temp = new ArrayList<>();
        List<TreeNode> helper = new ArrayList<>();  //设置一个临时存储下一层节点的局部变量
        while (!queue.isEmpty()) {
            TreeNode t = queue.poll();
            temp.add(t.val);
            //放入下一层节点
            if (t.left != null) helper.add(t.left);
            if (t.right != null) helper.add(t.right);
        }
        queue.addAll(helper);  //下一层节点放回队列
        res.add(temp);
    }
    return res;
}
```



# Leetcode121. 买卖股票的最佳时机

用两个局部变量：之前最小的股价，和目前的答案。

遍历整个数组，如果当前股价小于之前的最小股价，则之前最小股价修改为当前股价；并且要随时更新最大值。

```java
public int maxProfit(int[] prices) {
    if (prices == null || prices.length <= 1)
        return 0;
    
    int minDay = prices[0];
    int res = Integer.MIN_VALUE;
    
    for (int i = 1; i < prices.length; i++) {
        if (prices[i] < minDay)
            minDay = prices[i];
        res = Math.max(res, prices[i] - minDay);
    }
    
    return res;
}
```



# Leetcode160. 相交链表

首先判断两条链表是否有交点，即判断两条链表的最后一个节点是否是同一个节点。

如果没有交点即返回 null。

如果有交点则首先计算出两条链表的长度差，然后在长的链表先走长度差步，之后再两条链表一起走，直到走到交点。

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null)
        return null;
    ListNode curA = headA;
    ListNode curB = headB;
    // 判断链表有无交点
    while (curA.next != null) {
        curA = curA.next;
    }
    while (curB.next != null) {
        curB = curB.next;
    }
    if (curA != curB)
        return null;
    
    curA = headA;
    curB = headB;
    int lenA = 0;
    int lenB = 0;
    while (curA != null) {
        curA = curA.next;
        lenA ++;
    }
    while (curB != null) {
        curB = curB.next;
        lenB ++;
    }
    
    ListNode curLong;
    ListNode curShort;
    int diff = Math.abs(lenA - lenB);
    
    if (lenA > lenB) {
        curLong = headA;
        curShort = headB;
    } else {
        curLong = headB;
        curShort = headA;
    }
    
    for (int i = 0; i < diff; i++) {
        curLong = curLong.next;
    }
    
    while (curLong != curShort) {
        curLong = curLong.next;
        curShort = curShort.next;
    }
    
    return curShort;
}
```



# Leetcode88. 合并两个有序数组

题目要求时间复杂度为 O(m + n)，因此我开辟了一个数组用来存储，最后再把这个开辟的数组复制到 nums1 数组即可。

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    if (nums1 == null || nums2 == null)
        return;
    
    int[] nums = new int[m + n];
    int i = 0, j = 0, k = 0;
    
    while (i < m && j < n) {
        if (nums1[i] < nums2[j])
            nums[k ++] = nums1[i ++];
        else
            nums[k ++] = nums2[j ++];
    }
    while (i < m) {
        nums[k ++] = nums1[i ++];
    }
    while (j < n) {
        nums[k ++] = nums2[j ++];
    }
    
    if (m + n >= 0)
        System.arraycopy(nums, 0, nums1, 0, m + n);
}
```



# Leetcode20. 有效的括号

使用栈实现。

```java
public boolean isValid(String s) {
    if (s == null)
        return false;
    
    Stack<Character> stack = new Stack<>();
    
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        
        if (c == '(' || c == '[' || c == '{') {
            // 入栈
            stack.push(c);
        } else {
            // 出栈
            if (stack.size() == 0)
                return false;
            
            Character d = stack.peek();
            if (c == ')' && d != '(') {
                return false;
            }
            if (c == ']' && d != '[') {
                return false;
            }
            if (c == '}' && d != '{') {
                return false;
            }
            stack.pop();
        }
    }
    return stack.size() == 0;
}
```



# Leetcode103. Z字形层序遍历二叉树

可以用一个 flag 临时布尔变量判断当前是顺序打印还是逆序打印。

代码：

```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root == null)
        return res;
    
    Queue<TreeNode> deque = new ArrayDeque<>();
    deque.offer(root);
    
    boolean flag = true;
    while (!deque.isEmpty()) {
        List<Integer> temp = new ArrayList<>();
        List<TreeNode> helper = new ArrayList<>();
        
        if (flag) {
            while (!deque.isEmpty()) {
                TreeNode node = deque.poll();
                temp.add(node.val);
                if (node.left != null) helper.add(node.left);
                if (node.right != null) helper.add(node.right);
            }
        } else {
            while (!deque.isEmpty()) {
                TreeNode node = deque.poll();
                temp.add(node.val);
                
                // 注意这里，因为逆序打印，下一层顺序，因为先添加右节点再添加左节点。
                if (node.right != null) helper.add(node.right);
                if (node.left != null) helper.add(node.left);
            }
        }
        
        // 顺序打印时反转，因为下一层是逆序，打印按照逆序打印
        // 逆序打印时反转，因为下一层是顺序，因此需要逆序插入队列中
        Collections.reverse(helper);
        
        res.add(temp);
        deque.addAll(helper);
        flag = !flag;
    }
    return res;
}
```



# Leetcode235. 二叉搜索树的最近公共祖先

定义一个函数：`f(x)` 表示 `x` 节点的左子树或者右子树包含 `p` 或者 `q` 节点，或者 `x` 节点本身就是 `p` 或者 `q` 节点。

那么如何判断 `x` 是否是 `p` 或者 `q` 的最近公共祖先呢？

定义 `lson` 是 `f(x.left)`，`rson`  是 `f(x.right)` 

如果 `lson && rson` 为 `true`，那么 `x` 一定满足条件。

或者 `(x.val == p.val || x.val == q.val) && (lson || rson)`，如果 `x` 是 `p` 或者 `q` 并且左子树或者右子树有 `p` 节点或者 `q`节点。

代码：

```java
private TreeNode ans = null;

public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    dfs(root, p, q);
    return ans;
}

private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null) return false;
    boolean lson = dfs(root.left, p, q);
    boolean rson = dfs(root.right, p, q);
    if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
        ans = root;
    }
    
    return lson || rson || (root.val == p.val || root.val == q.val);
}
```



# Leetcode5. 最长回文子串

使用中心扩展法。

由回文串的特性，可以得知：
如果回文串的长度是奇数，那么两个指针指向回文串最中间的那个数可以向外扩展。
如果回文串的长度是偶数，那么两个指针分别指向中间左边的那个字符和中间右边的那个字符可以向外扩展。

我们可以遍历整个字符串，代码：

```java
public String longestPalindrome(String s) {
    if (s == null) return null;
    int n = s.length();
    if (n < 2)
        return s;
    String res = s.charAt(0) + "";
    
    for (int i = 0; i < n; i++) {
        String s1 = extend(i, i, s);  //奇数
        String s2 = extend(i, i + 1, s);  //偶数
        if (Math.max(s1.length(), s2.length()) > res.length()) {
            res = s1.length() > s2.length() ? s1 : s2;
        }
    }
    
    return res;
}

private String extend(int i, int j, String s) {
    while (i >=0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
        i --;
        j ++;
    }
    return s.substring(i + 1, j);
}
```



# Leetcode33. 搜索旋转数组

可以使用二分算法，首先判断 mid 指针指向的是较大的一段升序数组还是较小的一段升序数组。

判断的依据就是将 `nums[mid]` 和 `nums[0]` 比较，如果 `nums[mid]` 大说明 mid 指向的数在较大的一段升序数组中，那么我们就判断
`nums[0] < target && target < nums[mid]` 如果为真，说明 target 在第一段数组中，那么令 r = mid - 1，否则 l = mid + 1

如果 `nums[mid]` 小说明 mid 指向的数在较小的一段升序数组中，那么我们就判断`nums[mid] < target && target < nums[n - 1]` 如果为真，说明 target 在第二段数组中，那么令 l = mid + 1，否则 r = mid - 1;

```java
public int search(int[] nums, int target) {
    int n = nums.length;
    if (n == 0) {
        return -1;
    }
    
    if (n == 1) {
        return nums[0] == target ? 0 : -1;
    }
    
    int l = 0, r = n - 1;
    while (l <= r) {
        int mid = l + r >> 1;
        if (nums[mid] == target)
            return mid;
        
        if (nums[mid] >= nums[0]) {
            if (nums[0] <= target && target <= nums[mid]) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        } else {
            if (nums[mid] <= target && target <= nums[n - 1]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    
    }
    return -1;
}
```



# Leetcode200. 岛屿数量

这题使用 DFS算法，如果要求不改变原数组，就创建一个 st 判重数组，遍历过则置为 true

代码：

````java
private boolean[][] st;
public int numIslands(char[][] grid) {
    if (grid == null || grid.length == 0) {
        return 0;
    }
    
    int count = 0;
    int nr = grid.length;
    int nc = grid[0].length;
    
    st = new boolean[nr][nc];
    
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            // 这是岛屿并且没有被遍历过
            if (grid[i][j] == '1' && !st[i][j]) {
                count ++;
                dfs(i, j, grid);
            }
        }
    }
    
    return count;
}

private void dfs(int x, int y, char[][] grid) {
    if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length)
        return;
    // 这不是岛屿或则已经遍历过
    if (grid[x][y] != '1' || st[x][y])
        return;
    
    st[x][y] = true;
    
    dfs(x - 1, y, grid);
    dfs(x + 1, y, grid);
    dfs(x, y - 1, grid);
    dfs(x, y + 1, grid);
}
````



# Leetcode46. 全排列

这题用 Set 集合 + 回溯解决，模板题，牢记。

```java
private List<List<Integer>> ans;
private final Set<Integer> set = new HashSet<>();

public List<List<Integer>> permute(int[] nums) {
    if (nums == null)
        return null;
    ans = new ArrayList<>();
    if (nums.length == 0)
        return ans;
    
    dfs(0, nums, new ArrayList<>());
    return ans;
}

private void dfs(int index, int[] nums, ArrayList<Integer> path) {
    if (index == nums.length) {
        ans.add(new ArrayList<>(path));
        return;
    }
    for (int i = 0; i < nums.length; i++) {
        if (!set.contains(nums[i])) {
            set.add(nums[i]);
            path.add(nums[i]);
            dfs(index + 1, nums, path);
            set.remove(nums[i]);
            path.remove(path.size() - 1);
        }
    }
}
```



# LeetCode415. 字符串相加

其实就是大数相加，将两个字符串反转然后相加，注意进位处理即可。

```java
public String addStrings(String num1, String num2) {
    for (int i = 0; i < num1.length(); i++) {
        if (num1.charAt(i) < '0' || num1.charAt(i) > '9')
            return null;
    }
    for (int i = 0; i < num2.length(); i++) {
        if (num2.charAt(i) < '0' || num2.charAt(i) > '9')
            return null;
    }
    
    StringBuilder s1 = new StringBuilder(num1);
    StringBuilder s2 = new StringBuilder(num2);
    s1.reverse(); s2.reverse();
    
    StringBuilder res = new StringBuilder();
    int t = 0;  //存储进位
    int i = 0;  //存储下标
    for (i = 0; i < s1.length() && i < s2.length(); i++) {
        int sum = (s1.charAt(i) - '0') + (s2.charAt(i) - '0') + t;
        t = sum / 10;
        res.append(sum % 10);
    }
    
    if (i < s1.length()) {
        for (int j = i; j < s1.length(); j++) {
            int sum = s1.charAt(j) - '0' + t;
            t = sum / 10;
            res.append(sum % 10);
        }
        if (t != 0)
            res.append(t);
    } else if (i < s2.length()) {
        for (int j = i; j < s2.length(); j++) {
            int sum = s2.charAt(j) - '0' + t;
            t = sum / 10;
            res.append(sum % 10);
        }
        if (t != 0)
            res.append(t);
    } else if (t != 0) {
        res.append(t);
    }
    
    res.reverse();
    return res.toString();
}
```



# LeetCode92. 反转部分链表

我用了迭代的方法，首先反转部分链表，然后调整部分链表的前后指针指向。

代码：

```java
public ListNode reverseBetween(ListNode head, int left, int right) {
    if (left == right)
        return head;
    if (head == null)
        return null;
    
    ListNode dummy = new ListNode();
    dummy.next = head;
    // temp 指向部分链表的前一个节点， helper 指向部分链表的后一个节点
    // l 指向部分链表的最左侧，r 指向部分链表的最右侧
    // pre 用来帮助反转链表
    ListNode l, r, pre, temp, helper;
    l = head; r = head; temp = dummy;
    for (int i = 1; i < left; i++) {
        l = l.next;
        temp = temp.next;
    }
    for (int i = 1; i < right; i++) {
        r = r.next;
    }
    helper = r.next;
    
    // 反转链表
    pre = null;
    ListNode cur = l, nxt = l;
    while (cur != r.next) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    
    // 调整前后指针状态，达到真正的反转部分链表
    temp.next = r;
    l.next = helper;
    
    return dummy.next;
}
```



# Leetcode142. 环形链表

使用快慢指针，首先判断是否有环，如果无环返回 null，如果有环则随便让 fast 或者 slow 指针指向头结点，然后两个指针都已步长为1的步伐行走，最后相遇，返回相遇的那个节点。

```java
public ListNode detectCycle(ListNode head) {
    if (head == null)
        return null;
    
    boolean flag = false; // 判断是否有环，如果有环则为 true
    
    ListNode fast = head, slow = head;
    while (fast != null) {
        fast = fast.next;
        slow = slow.next;
        if (fast == null)
            break;
        fast = fast.next;
        if (fast == slow) {
            flag = true;
            break;
        }
    }
    
    // 无环
    if (!flag) return null;
    
    fast = head;
    while (fast != slow) {
        fast = fast.next;
        slow = slow.next;
    }
    
    return fast;
}
```



# Leetcode82. 合并 k 个有序链表

## 暴力算法

这题最暴力的算法就是两两合并链表

时间复杂度：O(k * n * k) 合并两个长度为 kn 的有序链表时间复杂度为 kn ，要合并 k 次，因此时间复杂度为 O(k^2^ x)

代码：

````java
public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
        return null;
    if (lists.length == 1)
        return lists[0];
    
    ListNode res = mergeTwoLists(lists[0], lists[1]);
    
    for (int i = 2; i < lists.length; i++) {
        res = mergeTwoLists(res, lists[i]);
    }
    return res;
}
// 合并两个链表
private ListNode mergeTwoLists(ListNode head1, ListNode head2) {
    ListNode dummy = new ListNode();
    ListNode p = dummy;
    ListNode l1 = head1, l2 = head2;
    
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            p.next = l1;
            l1 = l1.next;
        } else {
            p.next = l2;
            l2 = l2.next;
        }
        p = p.next;
    }
    
    if (l1 != null) {
        p.next = l1;
    }
    if (l2 != null) {
        p.next = l2;
    }
    
    return dummy.next;
}
````

## 分治法

<img src="C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220304113030710.png" alt="image-20220304113030710" style="zoom: 50%;" />

栈递归深度为 O(logk) ，时间复杂度为 O(k * n * logk)

代码：

```java
public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
        return null;
    if (lists.length == 1)
        return lists[0];
    
    return merge(lists, 0, lists.length - 1);
}

// 分治合并
private ListNode merge(ListNode[] lists, int l, int r) {
    if (l == r) {
        return lists[l];
    }
    if (l > r) {
        return null;
    }
    int mid = l + r >> 1;
    return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 
}

// 合并两个有序链表
private ListNode mergeTwoLists(ListNode head1, ListNode head2) {
    if (head1 == null || head2 == null) {
        return head1 == null ? head2 : head1;
    }
    
    ListNode dummy = new ListNode();
    ListNode p = dummy;
    ListNode l1 = head1, l2 = head2;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            p.next = l1;
            l1 = l1.next;
        } else {
            p.next = l2;
            l2 = l2.next;
        }
        p = p.next;
    }
    
    if (l1 != null) {
        p.next = l1;
    }
    if (l2 != null) {
        p.next = l2;
    }
    
    return dummy.next;
}
```

## 优先队列

使用一个优先队列维护节点，优先队列队头一定是当前节点中最小的数，我们可以将链表数组的第一个节点放入优先队列中，并将优先队列中最小的数取出接到链表中，然后放入接入的链表节点的下一个链表节点。

代码：

```java
// 优先队列维护的元素
class Status implements Comparable<Status> {
    int val;
    ListNode pointer;
    @Override
    public int compareTo(Status status) {
        return this.val - status.val;
    }
    Status(int val, ListNode ptr) {
        this.val = val;
        this.pointer = ptr;
    }
}

private final PriorityQueue<Status> pile = new PriorityQueue<>();

public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
        return null;
    if (lists.length == 1)
        return lists[0];
    
    for (ListNode list : lists) {
        if (list != null)
            pile.offer(new Status(list.val, list));
    }
    
    ListNode dummy = new ListNode();
    ListNode cur = dummy;
    
    while (!pile.isEmpty()) {
        Status peek = pile.poll();
        cur.next = peek.pointer;
        cur = cur.next;
        
        if (peek.pointer.next != null) {
            pile.offer(new Status(peek.pointer.next.val, peek.pointer.next));
        }
    }
    
    return dummy.next;
}
```



# Leetcode54. 螺旋矩阵

模拟题，公式：

<img src="C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220306095722233.png" alt="image-20220306095722233" style="zoom: 33%;" />

注意如果用迭代可能导致会加入不必要的数字，因此在 add 方法应该加一个 if 判断

```java
public List<Integer> spiralOrder(int[][] matrix) {
    if (matrix == null)
        return null;
    List<Integer> list = new ArrayList<>();
    int m = matrix.length;
    int n = matrix[0].length;
    
    if (m == 1) {
        for (int i = 0; i < n; i++) {
            list.add(matrix[0][i]);
        }
        return list;
    }
    
    if (n == 1) {
        for (int i = 0; i < m; i++) {
            list.add(matrix[i][0]);
        }
        return list;
    }
    
    for (int i = 0; i < Math.min(m, n); i++) {
        for (int index1 = i; index1 <= n - i - 1; index1++) {
            if (list.size() < m * n)
                list.add(matrix[i][index1]);
            else return list;
        }
        for (int index2 = i + 1; index2 <= m - i - 1; index2++) {
            if (list.size() < m * n)
                list.add(matrix[index2][n - 1 - i]);
            else return list;
        }
        for (int index3 = n - i - 2; index3 >= i; index3--) {
            if (list.size() < m * n)
                list.add(matrix[m - 1 - i][index3]);
            else return list;
        }
        for (int index4 = m - 2 - i; index4 >= i + 1; index4--) {
            if (list.size() < m * n)
                list.add(matrix[index4][i]);
            else return list;
        }
    }
    return list;
}
```



# Leetcode300. 最长上升子序列

## DP做法

`dp[i]` 表示以 `nums[i]` 结尾的最长上升子序列长度

状态转移方程： `dp[i] = max{dp[j]} + 1， nums[j] < nums[i] && j < i`

代码：

```java
public int lengthOfLIS(int[] nums) {
    if (nums == null) return 0;
    int n = nums.length;
    if (n <= 1) return n;
    
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = Math.max(dp[j] + 1, dp[i]);
            }
        }
    }
    
    int res = 0;
    for (int i : dp) {
        res = Math.max(res, i);
    }
    return res;
}
```

## 打印路径

我们需要开辟一个数组 path，其中如果 `path[i] = j` 表示在某一个 LIS 中下标为 j 的元素的前一个元素是 nums 数组中下标为 i 的元素

代码：

````java
static int lenthOfLIS(int[] nums) {
    if (nums == null) return 0;
    int n = nums.length;
    if (n <= 1) return n;
    
    List<Integer> way = new ArrayList<>();
    
    int[] dp = new int[n];
    int[] path = new int[n];
    Arrays.fill(path, -1);
    Arrays.fill(dp, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                //dp[i] = Math.max(dp[j] + 1, dp[i]);
                if (dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    // 记录 i 的前一个下标 j
                    path[i] = j;
                }
            }
        }
    }
    int res = 0, k = -1;
    for (int i = 0; i < n; i++) {
        if (res < dp[i]) {
            res = dp[i];
            k = i;  // 找到 LIS 的最后一个数的下标
        }
    }
    // 逆序填入 way 数组
    while (k >= 0) {
        way.add(nums[k]);
        k = path[k];
    }
    Collections.reverse(way);
    System.out.println(way);
    
    return res;
}
````

## nlogn 时间复杂度算法（DP + 二分）

用 tails 数组优化算法：

tails数组含义：`tails[k] 的值代表 长度为 k+1 子序列 的尾部元素值。`

**转移方程**： 设 res 为 tails 当前长度，代表直到当前的最长上升子序列长度。设 j∈[0,res)，考虑每轮遍历nums[k] 时，通过二分法遍历 [0,res) 列表区间，找出 nums[k] 的大小分界点，会出现两种情况：

- **区间中存在 tails[i] > nums[k]** ： 将第一个满足 tails[i]>nums[k] 执行 tails[i]=nums[k] ；因为更小的 nums[k] 后更可能接一个比它大的数字。
- **区间中不存在 tails[i] > nums[k]** ： 意味着 nums[k] 可以接在前面所有长度的子序列之后，因此肯定是接到最长的后面（长度为 res ），新子序列长度为 res + 1。

代码：

````java
public int lengthOfLIS(int[] nums) {
    if (nums == null) return 0;
    int n = nums.length;
    if (n <= 1) return n;
    
    int[] tails = new int[nums.length];
    int res = 0;
    for(int num : nums) {
        int i = 0, j = res;
        while(i < j) {
            int m = (i + j) / 2;
            if(tails[m] < num) i = m + 1;
            else j = m;
        }
        tails[i] = num;
        if(res == j) res++;
    }
    return res;
}
````

# Leetcode42. 接雨水

这题可以用 DP 做也可以用单调栈做，DP 的做法更好懂一些。

开辟两个数组：

`leftMax[n]`：`leftMax[i]` 表示 `height[0..i]` 中最大的 `height` 

`rightMax[n]`：`rightMax[i]` 表示 `height[i..n-1]` 中最大的 `height` 

代码：

```java
public int trap(int[] height) {
    if (height == null || height.length <= 1) return 0;
    int res = 0;
    int n = height.length;
    
    int[] leftMax = new int[n];
    int[] rightMax = new int[n];
    leftMax[0] = height[0];
    for (int i = 1; i < n; i++) {
        leftMax[i] = Math.max(leftMax[i - 1], height[i]);
    }
    
    rightMax[n - 1] = height[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        rightMax[i] = Math.max(rightMax[i + 1], height[i]);
    }
    
    for (int i = 0; i < n; i++) {
        res += Math.min(leftMax[i], rightMax[i]) - height[i];
    }
    return res;
}
```

<img src="C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220309144257094.png" alt="image-20220309144257094" style="zoom:67%;" />

# Leetcode143. 重排链表

题目要求对链表进行重排序，我用一个数组将这些引用存下来，用数组虽然有O(n)的时间复杂度，但是也方便我去对链表的随机访问。

最终复杂度：时间复杂度为O(n)，空间复杂度为O(n)

代码：

```java
public void reorderList(ListNode head) {
    if (head == null) return;
    
    int size = 0;
    ListNode p = head;
    while (p != null) {
        size ++;
        p = p.next;
    }
    
    p = head;
    ListNode[] arr = new ListNode[size];
    for (int i = 0; i < size; i++) {
        arr[i] = p;
        p = p.next;
    }
    
    int l = 0, r = size - 1;
    while (l < r) {
        arr[l].next = arr[r];
        
        if (r - 1 == l + 1) {
            // 链表个数为奇数情况
            arr[l].next = arr[r];
            arr[r].next = arr[l + 1];
            arr[l + 1].next = null;
            break;
        }
 		// 链表个数为偶数情况   
        if (r != l + 1)
            arr[r--].next = arr[++l];
        else {
            arr[r].next = null;
            break;
        }
    }
}
```

# Leetcode199. 二叉树的右视图

要求打印二叉树的右视图。

使用 BFS 算法，存储每一层的节点，最后打印每一层的最后一个节点即可。

代码：

```java
List<Integer> res = new ArrayList<>();

public List<Integer> rightSideView(TreeNode root) {
    if (root == null) return res;
    bfs(root);
    return res;
}

private void bfs(TreeNode root) {
    List<List<Integer>> path = new ArrayList<>();
    Queue<TreeNode> q = new ArrayDeque<>();
    q.add(root);

    while (!q.isEmpty()) {
        List<Integer> temp = new ArrayList<>();
        List<TreeNode> helper = new ArrayList<>();
        while (!q.isEmpty()) {
            TreeNode t = q.poll();
            temp.add(t.val);
            if (t.left != null)
                helper.add(t.left);
            if (t.right != null)
                helper.add(t.right);
        }
        path.add(temp);
        q.addAll(helper);
    }
 
    for (List<Integer> integers : path) {
        res.add(integers.get(integers.size() - 1));
    }
}
```

# Leetcode124. 二叉树的最大路径和

这题需要使用递归来写。

写一个函数：`dfs(root)` 这个函数的作用是计算 root 所在的数的最大子树和（包括root节点）

代码：

```java
int maxSum = Integer.MIN_VALUE;
public int maxPathSum(TreeNode root) {
    if (root == null) return -1;
    dfs(root);
    return maxSum;
}

private int dfs(TreeNode root) {
    if (root == null) return 0;
    int l = Math.max(dfs(root.left), 0);
    int r = Math.max(dfs(root.right), 0);
    maxSum = Math.max(maxSum, l + r + root.val);
    return Math.max(l, r) + root.val;
}
```

# Leetcode70. 爬楼梯

实际上就是求斐波那契数列，由于当前状态只依赖前一个状态和后一个状态，因此可以优化空间复杂度到O(1)

代码：

```java
    public int climbStairs(int n) {
        if (n <= 2) return n;
        int preStep = 1;
        int curStep = 2;
        for (int i = 3; i <= n; i++) {
            int temp = curStep;
            curStep = curStep + preStep;
            preStep = temp;

        }
        return curStep;
    }
```

# Leetcode56. 合并区间

思路大致就是首先按照区间左侧排序，排序完成后合并即可（用局部变量 leftBound 和 rightBound）

```java
public int[][] merge(int[][] intervals) {
    if (intervals == null) return null;
    
    List<int[]> res = new ArrayList<>();
    Arrays.sort(intervals, (o1, o2) -> Integer.compare(o1[0], o2[0]));
    int leftBound = intervals[0][0]; int rightBound = intervals[0][1];
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] > rightBound) {
            res.add(new int[]{leftBound, rightBound});
            leftBound = intervals[i][0];
            rightBound = intervals[i][1];
        } else {
            rightBound = Math.max(rightBound, intervals[i][1]);
        }
    }
    res.add(new int[]{leftBound, rightBound});
    
    int[][] ans = new int[res.size()][2];
    for (int i = 0; i < res.size(); i++) {
        ans[i] = res.get(i);
    }
    return ans;
}
```

# 剑指Offer22. 链表中倒数第 k 个节点

首先计算链表的长度，那么步长 `step = size - k`

代码：

```java
public ListNode getKthFromEnd(ListNode head, int k) {
    if (head == null) return null;
    
    int size = 0;
    ListNode cur = head;
    while (cur != null) {
        cur = cur.next;
        size ++;
    }
    
    int step = size - k;
    cur = head;
    for (int i = 0; i < step; i++) {
        cur = cur.next;
    }
    
    return cur;
}
```

# Leetcode19. 删除链表的倒数第 n 个节点

和上一题一样，首先计算出步长，然后设置一个pre指针指向cur指针的前一个节点以便删除cur节点。

代码：

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    if (head == null) return null;
    
    int size = 0;
    ListNode cur = head;
    while (cur != null) {
        cur = cur.next;
        size ++;
    }
    int step = size - n;
    
    ListNode dummy = new ListNode();
    dummy.next = head;
    
    ListNode pre = dummy;
    cur = head;
    
    for (int i = 0; i < step; i++) {
        pre = pre.next;
        cur = cur.next;
    }
    pre.next = cur.next;
    
    return dummy.next;
}
```



# Leetcode69. x 的平方根

用 二分法

代码：

```java
public int mySqrt(int x) {
    long l = 0, r = x;
    while (l < r) {
        long mid = l + r + 1 >> 1;
        if (mid * mid <= x)
            l = mid;
        else
            r = mid - 1;
    }
    return (int) l;
}
```

二分模板（找最小值中的最大值与最大值中的最小值）：

**找最小值中的最大值**：

```java
int main()
{
	int l;
	int r;
	while(l < r)
	{
		int mid = (l + r + 1)/ 2;  // 这里要 l + r +1 要不然会死循环
		if(check())
		{
			l = mid;         // mid这个位置 满足条件之后 查找 [mid , right]的位置， 所以l移到mid的位置
		}
		else
		{
			r = mid - 1;     // [mid,r] 不满足条件， 所以要移到满足条件的一方， r = mid - 1 
		}
	} 
	//最后的l,r是答案 因为 l == r

}
```

**找最大值中的最小值**：

```java
int main()
{
	int l;
	int r;
	while(l < r)
	{
		int mid = (l + r)/ 2;
		if(check())
		{
			r = mid;  // 这里是 r = mid, 说明[l,mid]是合法范围
		}
		else
		{
			l = mid + 1;   //  [l,mid]这个范围都不是合法范围，所以下一次查找直接从 l = mid + 1开始了
		}
		//最后的l,r是答案 因为 l == r ，最终就是答案。
	} 		
}
```

# Leetcode8. Atoi 字符串转换为整数

纯纯的模拟

代码：

```java
public int myAtoi(String s) {
    if (s == null) return 0;
    if (s.length() == 0) return 0;
    
    int idx = 0;
    for (idx = 0; idx < s.length(); idx++) {
        if (s.charAt(idx) != ' ')
            break;
    }
    
    boolean isSub = false;
    
    if (idx >= s.length()) return 0;  
    
    if (s.charAt(idx) == '-') {
        isSub = true;
        idx ++;
    } else if (s.charAt(idx) == '+') {
        idx ++;
    }
    
    if (idx >= s.length()) return 0;
    
    int start = idx;
    int end = start;
    for (int i = start; i < s.length(); i++) {
        if (s.charAt(i) < '0' || s.charAt(i) > '9') {
            end = i - 1;
            break;
        }
        end = i;
    }
    
    if (end < start) return 0;
    String number = s.substring(start, end + 1);
    return core(number, isSub);
}

private int core(String number, boolean flag) {
    long res = 0;
    int idx = 0;
    
    // 从非 0 开始
    for (idx = 0; idx < number.length(); idx++) {
        if (number.charAt(idx) != '0')
            break;
    }
    
    if (flag) {
        for (int i = idx; i < number.length(); i++) {
           // if (number.charAt(i) == '0') continue;
            res = res * 10 + (number.charAt(i) - '0');
            if (res >= (Integer.MAX_VALUE + 1L)) return Integer.MIN_VALUE;
        }
        res = -res;
    } else {
        // 正数
        for (int i = idx; i < number.length(); i++) {
           // if (number.charAt(i) == '0') continue;
            res = res * 10 + (number.charAt(i) - '0');
            if (res > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        }
    }
    
    return (int) res;
}
```

# Leetcode82. 删除链表的重复元素②

这题要注意排序的条件，不要用 Map。

代码：

```java
public ListNode deleteDuplicates(ListNode head) {
    if (head == null) return head;
    ListNode cur = head;
    ListNode dummy = new ListNode(); dummy.next = head;
    ListNode pre = dummy;
    ListNode nxt = head.next;
    if (nxt == null) return head;
    
    while (nxt != null) {
        if (cur.val == nxt.val) {
            int key = cur.val;
            while (cur != null && cur.val == key) {
                cur = cur.next;
            }
            if (cur == null) {
                pre.next = null;
                break;
            }
            nxt = cur.next;
            pre.next = cur;
        } else {
            pre = cur;
            cur = nxt;
            nxt = nxt.next;
        }
    }
    return dummy.next;
}
```

# Leetcode82. 排序链表

## 归并写法

```java
public ListNode sortList(ListNode head) {
    return sort(head, null);
}

// [head, tail) 对这个区间的链表进行排序
private ListNode sort(ListNode head, ListNode tail) {
    if (head == null) return null;
    
    if (head.next == tail) {
        // 链表只包含一个节点
        head.next = null;
        return head;
    }
    
    // 快慢指针找中点
    ListNode slow = head, fast = head;
    while (fast != tail) {
        slow = slow.next;
        fast = fast.next;
        if (fast != tail)
            fast = fast.next;
    }
    ListNode mid = slow;
    
    // 排序左边链表
    ListNode list1 = sort(head, mid);
    // 排序右边链表
    ListNode list2 = sort(mid, tail);
    // 合并
    ListNode sorted = merge(list1, list2);
    return sorted;
}

// 合并两个有序链表
private ListNode merge(ListNode list1, ListNode list2) {
    ListNode dummy = new ListNode();
    ListNode cur = dummy;
    ListNode l1 = list1, l2 = list2;
    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            cur.next = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    if (l1 != null) cur.next = l1;
    if (l2 != null) cur.next = l2;
    return dummy.next;
}
```

## 迭代写法

![image-20220314140039316](C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220314140039316.png)

```java
public ListNode sortList(ListNode head) {
    if (head == null) {
        return head;
    }
    
    // 获得链表长度
    int length = 0;
    ListNode node = head;
    while (node != null) {
        length++;
        node = node.next;
    }
    
    ListNode dummyHead = new ListNode(0, head);
    
    for (int subLength = 1; subLength < length; subLength <<= 1) {
        ListNode prev = dummyHead, curr = dummyHead.next;
        
        while (curr != null) {
            
            // 找到第一段链表
            ListNode head1 = curr;
            for (int i = 1; i < subLength && curr.next != null; i++) {
                curr = curr.next;
            }
            
            // 找到第二段链表
            ListNode head2 = curr.next;
            curr.next = null;  // 断开第一段链表
            curr = head2;
            for (int i = 1; i < subLength && curr != null && curr.next != null; i++
                curr = curr.next;
            }
            
            // 下一次选择 head1 和 head2 的开头
            ListNode next = null;
            if (curr != null) {
                next = curr.next;
                curr.next = null;  // 断开第二段链表
            }
            
            // 将第一段和第二段链表合并
            prev.next = merge(head1, head2);
            while (prev.next != null) {
                prev = prev.next;
            }
            curr = next;
        }
    }
    return dummyHead.next;
}
// 合并有序链表
public ListNode merge(ListNode head1, ListNode head2) {
    
    ListNode dummyHead = new ListNode(0);
    ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
    
    while (temp1 != null && temp2 != null) {
        if (temp1.val <= temp2.val) {
            temp.next = temp1;
            temp1 = temp1.next;
        } else {
            temp.next = temp2;
            temp2 = temp2.next;
        }
        temp = temp.next;
    }
    
    if (temp1 != null) {
        temp.next = temp1;
    } else if (temp2 != null) {
        temp.next = temp2;
    }
    return dummyHead.next;
}
```



# Leetcode72. 编辑距离

动态规划问题，操作分为 插入、删除、修改操作

像这种两种字符串的问题一般创建二维 dp 数组

`dp[i][j]` 表示字符串1[0 .. (i-1)] 子串修改为 字符串二[0 .. (j-1)] 子串最少的编辑距离为多少

状态转移方程：

```java
			dp[i - 1][j - 1],	word1[i] = word2[j]
dp[i][j] = 
			min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1,	word1[i] != word2[j]
			// 删除操作			替换操作			插入操作
```

初始化：

```java
dp[0][0..l2] = i;
dp[0..l1][0] = i;
```



代码：

```java
public int minDistance(String word1, String word2) {
    if (word1 == null) {
        if (word2 == null) {
            return 0;
        } else {
            return word2.length();
        }
    }
    if (word2 == null) {
        return word1.length();
    }
    int l1 = word1.length();
    int l2 = word2.length();
    int[][] dp = new int[l1 + 1][l2 + 1];
    for (int i = 1; i <= l2; i++) {
        dp[0][i] = i;
    }
    for (int i = 1; i <= l1; i++) {
        dp[i][0] = i;
    }
    for (int i = 1; i <= l1; i++) {
        for (int j = 1; j <= l2; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1))
                dp[i][j] = dp[i - 1][j - 1];
            else {
                dp[i][j] = min(
                        dp[i - 1][j] + 1,   // 删除操作
                        dp[i - 1][j - 1] + 1,   // 替换操作
                        dp[i][j - 1] + 1    // 插入操作
                );
            }
        }
    }
    return dp[l1][l2];
}
private int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}
```

# Leetcode2. 两数相加

如果可以改变原来链表我们就在原来的链表上操作，如果不能改变原来的链表我们只好 new 一条新的链表处理了。

代码：

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode();
    ListNode cur = dummy;
    ListNode p1 = l1;
    ListNode p2 = l2;
    
    int t = 0;
    while (p1 != null && p2 != null) {
        int temp = p1.val;
        p1.val = (p1.val + p2.val + t) % 10;
        t = (temp + p2.val + t) / 10;
        cur.next = p1;
        p1 = p1.next;
        p2 = p2.next;
        cur = cur.next;
    }
    
    while (p1 != null) {
        int temp = p1.val;
        p1.val = (p1.val + t) % 10;
        t = (temp + t) / 10;
        cur.next = p1;
        p1 = p1.next;
        cur = cur.next;
    }
    
    while (p2 != null) {
        int temp = p2.val;
        p2.val = (p2.val + t) % 10;
        t = (temp + t) / 10;
        cur.next = p2;
        p2 = p2.next;
        cur = cur.next;
    }
    
    if (t != 0) {
        cur.next = new ListNode(t, null);
    }
    return dummy.next;
}
```



# Leetcode4. 寻找两个正序数组中的中位数

这题可以用二分的思想

寻找两个正序数组中的中位数，转为寻找两个数组的第 k 个小的数

因为这两个数组都是有序的，我们要充分利用有序的条件

如果两个数组长度和为奇数 那就寻找 k = total / 2

如果两个数组长度和为偶数，那就寻找 k = total / 2 - 1 和 k = total / 2 这两个最小数

在寻找两个数组中第 k 个最小数时

我们又可以找到两个数组的 k / 2 - 1 索引对应的数

如果 num1 < num2 那么 nums1 数组中的前 k / 2 - 1 个数可以全部舍去，因为这些舍去的数一定在两个数组第 k 个最小数中，然后更新索引，进行下一轮

num2 同理

```java
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + 
                             getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。
         * 把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。
         * 把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;

        while (true) {
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            int newIndex1 = Math.min(length1, index1 + k / 2) - 1;
            int newIndex2 = Math.min(length2, index2 + k / 2) - 1;
            int num1 = nums1[newIndex1], num2 = nums2[newIndex2];
            if (num1 <= num2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }
```



# Leetcode144.  二叉树前序遍历

## 递归写法

默写之。

```java
private final List<Integer> ans = new ArrayList<>();

public List<Integer> preorderTraversal(TreeNode root) {
    dfs(root);
    return ans;
}

private void dfs(TreeNode root) {
    if (root == null) return;
    ans.add(root.val);
    dfs(root.left);
    dfs(root.right);
}
```

## 迭代写法



```java
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) {
        return res;
    }
    
    Stack<TreeNode> stack = new Stack<>();
    TreeNode node = root;
    while (!stack.isEmpty() || node != null) {
        
        while (node != null) {
            res.add(node.val);
            stack.push(node);
            node = node.left;
        }
        
        node = stack.pop();
        node = node.right;
    }
    
    return res;
}
```



# Leetcode104. 二叉树的最大深度

## DFS

```java
private int dfs(TreeNode root) {
    if (root == null) return 0;
    int maxLeft = dfs(root.left);
    int maxRight = dfs(root.right);
    return Math.max(maxLeft, maxRight) + 1;
}
```



## BFS

```java
private int bfs(TreeNode root) {
    if (root == null) return 0;
    int maxDepth = 0;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
        int sz = q.size();
        while (sz > 0) {
            TreeNode t = q.poll();
            if (t.left != null) q.offer(t.left);
            if (t.right != null) q.offer(t.right);
            sz --;
        }
        maxDepth ++;
    }
    return maxDepth;
}
```



# Leetcode93. 复原 IP 地址

思路：

先对数字序列可能添加 . 的位置进行全排列，找到三个位置之后对字符串一一判断即可。

判断依据为数字的大小是否大于 255，以及是否以 0 开头且位数大于 1 位

代码：

```java
class Solution {
    
    List<String> ans = new ArrayList<>();
    
    String str = null;
    
    public List<String> restoreIpAddresses(String s) {
        if (s.length() > 12) return ans;
        str = s;
        dfs(s.length() - 1, 0, new ArrayList<>());
        return ans;
    }

    private void dfs(int length, int index, List<Integer> path) {
        if (path.size() == 3 || index >= length) {
            if (path.size() == 3) {
                doCheck(path);
            }
            return;
        }
        
        path.add(index);
        dfs(length, index + 1, path);
        path.remove(path.size() - 1);

        dfs(length, index + 1, path);
    }

    private void doCheck(List<Integer> path) {
        String num1 = str.substring(0, path.get(0) + 1);
        String num2 = str.substring(path.get(0) + 1, path.get(1) + 1);
        String num3 = str.substring(path.get(1) + 1, path.get(2) + 1);
        String num4 = str.substring(path.get(2) + 1);
        for (String s : Arrays.asList(num1, num2, num3, num4)) {
            if (Long.parseLong(s) > 255) return;
            if (s.startsWith("0") && s.length() > 1) return;
        }
        ans.add(num1 + "." + num2 + "." + num3 + "." + num4);
    }
}
```

 

# Leetcode41. 缺失的第一个正整数

给一个数组，找出其中缺失的第一个正整数。

要求时间复杂度为O(n)，空间复杂度为 O(1)

这题如果想空间复杂度 O(1)，则必须改动原数组。

我们使用 原地 Hash 算法

长度为 n 的数组其中缺失的第一个正整数范围只能在[1, n + 1]

遍历一遍数组，将小于等于0的数置为n + 1

再遍历一遍数组，将绝对值小于等于 n 的元素 num

```java
nums[num - 1] = -Math.abs(nums[num - 1]);  // 置为绝对负数
```

最后遍历一遍数组值为正数的元素对应下标加1即为答案，如果数组中没有正数则返回 n + 1

代码：

```java
public int firstMissingPositive(int[] nums) {
    if (nums == null || nums.length == 0) return 1;
    int n = nums.length;
    
    for (int i = 0; i < n; i++) {
        if (nums[i] <= 0) nums[i] = n + 1;
    }
    
    for (int i = 0; i < n; i++) {
        int num = Math.abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -Math.abs(nums[num - 1]);
        }
    }
    
    for (int i = 0; i < n; i++) {
        if (nums[i] >= 0) return i + 1;
    }
    return n + 1;
}
```



# Leetcode22. 括号生成

这题使用递归，递归函数状态：recur(String s, int left, int right)

s 表示当前拼接的字符串状态，left 表示剩余的左括号数量， right 表示剩余的有括号数量

不难发现规律：左括号的数量无论是结果还是在求解的过程中都必须满足大于等于右扩号数量

代码：

```java
    List<String> res = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        getParenthesis("", n, n);
        return res;
    }

    private void getParenthesis(String s, int left, int right) {
        if (left == 0 && right == 0) {
            res.add(s);
            return;
        }

        if (left == right) {
            // 左括号数量等于右括号数量
            // 那么只能添加左括号
            getParenthesis(s + "(", left - 1, right);
        } else if (left < right) {
            // 剩余的左括号数量少于剩余的右括号数量
            // 可以条件左括号也可以添加右括号
            if (left > 0)
                // 注意条件，否则会一直递归导致内存溢出
                getParenthesis(s + "(", left - 1, right);
            getParenthesis(s + ")", left, right - 1);
        }
    }
```



# Leetcode105. 已知前序和中序重建二叉树

经典算法，递归即可

代码：

```java
private Map<Integer, Integer> indexMap;

public TreeNode buildTree(int[] preorder, int[] inorder) {
    int n = preorder.length;
    indexMap = new HashMap<>();
    
    for (int i = 0; i < n; i++) {
        indexMap.put(inorder[i], i);
    }
    
    return myBuildTree(preorder, 0, n - 1, 0, n - 1);
}

public TreeNode myBuildTree(int[] preorder, int pl, int pr, int il, int ir) {
    if (pl > pr) return null;
    if (il > ir) return null;
    
    // 根节点在中序数组中的下标
    int k = indexMap.get(preorder[pl]);
    TreeNode root = new TreeNode(preorder[pl]);
    
    root.left = myBuildTree(preorder, pl + 1, pl + k - il, il, k - 1);
    root.right = myBuildTree(preorder, pl + k - il + 1, pr, k + 1, ir);
    return root;
}
```



# Leetcode98. 验证二叉搜索树

## 递归写法

这题要求是左子树的所有节点均小于根节点的值，右子树的所有节点均大于根节点的值，因此我们可以用递归来写。

```java
public boolean isValidBST(TreeNode root) {
    return dfs(root, Integer.MAX_VALUE + 1L, Integer.MIN_VALUE - 1L);
}

// 范围为(min, max) 如果 root 节点的值超过了这个范围，那么说明错误
private boolean dfs(TreeNode root, long max, long min) {
    if (root == null) {
        return true;
    }
    if (root.val >= max || root.val <= min) {
        return false;
    }
    return dfs(root.left, root.val, min) && dfs(root.right, max, root.val);
}
```



## 中序遍历

由于这是一颗二叉搜索树，因此它的中序遍历一定是有序的，我们每次存储它的前一个中序遍历节点的值即可。

代码：

```java
private Long pre = Integer.MIN_VALUE - 1L;

public boolean isValidBST(TreeNode root) {
    return inorder(root);
}

private boolean inorder(TreeNode root) {
    if (root == null) return true;
    
    boolean l = inorder(root.left);
    
    if (root.val <= pre) return false;
    pre = (long) root.val;
    
    boolean r = inorder(root.right);
    return l && r;
}
```



迭代版本：

使用栈模拟递归过程。

代码：

```java
public boolean isValidBST(TreeNode root) {
    Stack<TreeNode> stack = new Stack<>();
    if (root == null) return true;
    
    Long pre = (long) Integer.MIN_VALUE - 1;
    while (!stack.isEmpty() || root != null) {
        
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
        root = stack.pop();
        
        if (root.val <= pre) return false;
        pre = (long) root.val;
        
        root = root.right;
    }
    return true;
}
```

# Leetcode234. 回文链表

首先使用快慢指针找到链表中点，同时反转前半条链表，反转后再遍历比较即可。

代码：

```java
public boolean isPalindrome(ListNode head) {
    ListNode pre = null;
    ListNode slow = head;
    ListNode fast = head;
    
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        ListNode nxt = slow.next;
        slow.next = pre;
        pre = slow;
        slow = nxt;
    }
    
    if (fast != null) {
        // 奇数个节点
        slow = slow.next;
    }
    
    while (pre != null && slow != null) {
        if (pre.val != slow.val) return false;
        pre = pre.next;
        slow = slow.next;
    }
    return true;
}
```



# Leetcode155. 最小栈

用一个栈存储数据，一个栈作为辅助。

加入元素时，如果加入的元素小于辅助栈栈顶元素，则重复放入辅助栈栈顶元素，否则加入当前要加入的元素。

代码：

```java
private final Stack<Integer> data;
private final Stack<Integer> helper;

public MinStack() {
    data = new Stack<>();
    helper = new Stack<>();
}

public void push(int val) {
    data.push(val);
    if (helper.size() == 0) {
        helper.add(val);
        return;
    }
    if (helper.peek() < val) {
        helper.add(helper.peek());
    } else {
        helper.add(val);
    }
}

public void pop() {
    data.pop();
    helper.pop();
}

public int top() {
    return data.peek();
}

public int getMin() {
    return helper.peek();
}
```



# Leetcode78. 子集

回溯

```java
private final List<List<Integer>> res = new ArrayList<>();

public List<List<Integer>> subsets(int[] nums) {
    res.clear();
    dfs(nums, 0, new ArrayList<>());
    return res;
}

private void dfs(int[] nums, int i, List<Integer> path) {
    if (i == nums.length) {
        res.add(new ArrayList<>(path));
        return;
    }
    
    // 放入元素
    path.add(nums[i]);
    dfs(nums, i + 1, path);
    path.remove(path.size() - 1);
    
    // 不放入元素
    dfs(nums, i + 1, path);
}
```



# Leetcode239. 滑动窗口最大值

使用单调队列

每次放入数据时都和队尾比较，如果大于队尾元素则移出队尾元素直到队列空或者队尾元素大于当前元素。

代码：

````java
public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    int[] ans = new int[n - k + 1];
    Deque<Integer> queue = new LinkedList<>();
    
    for (int i = 0; i < k; i++) {
        while (!queue.isEmpty() && nums[i] >= nums[queue.getLast()]) {
            queue.removeLast();
        }
        queue.addLast(i);
    }
    ans[0] = nums[queue.getFirst()];
    
    for (int i = k; i < n; i++) {
        while (!queue.isEmpty() && nums[i] >= nums[queue.getLast()]) {
            queue.removeLast();
        }
        queue.addLast(i);
        // 当前队列长度大于 k 
        while (queue.getFirst() <= i - k) {
            queue.removeFirst();
        }
        ans[i - k + 1] = nums[queue.getFirst()];
    }
    return ans;
}
````



# Leetcode32. 最长连续括号

用栈来实现，当为 ( 时入栈，当为 ) 时出栈，判断栈是否为空，如果为空，则填入当前下标，否则更新答案

代码：

```java
public int longestValidParentheses(String s) {
    Stack<Integer> stack = new Stack<>();
    stack.add(-1);
    int maxAns = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            stack.push(i);
        } else {
            stack.pop();
            if (stack.isEmpty()) {
                stack.push(i);
            } else {
                maxAns = Math.max(maxAns, i - stack.peek());
            }
        }
    }
    return maxAns;
}
```



# Leetcode101. 二叉树的对称性

## 递归

使用递归判断二叉树是否镜像对称

两颗棵树是否镜像可以这样看：

- 它们根节点的值是否相等
- 每棵树的右子树都和另一棵树的左子树相等

代码：

```java
public boolean isSymmetric(TreeNode root) {
    return check(root, root);
}

private boolean check(TreeNode q, TreeNode p) {
    if (q == null && p == null)
        return true;
    if (q == null || p == null)
        return false;
    
    return q.val == p.val && check(q.left, p.right) && check(q.right, p.left);
}
```

## 迭代

使用队列迭代

代码：

```java
public boolean isSymmetric(TreeNode root) {
    return check_Q(root, root);
}

private boolean check_Q(TreeNode q, TreeNode p) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(q);
    queue.offer(p);
    
    while (!queue.isEmpty()) {
        TreeNode u = queue.poll();
        TreeNode v = queue.poll();
        
        if (u == null && v == null) {
            continue;
        }
        if (u == null || v == null || u.val != v.val) {
            return false;
        }
        
        queue.offer(u.left);
        queue.offer(v.right);
        
        queue.offer(u.right);
        queue.offer(v.left);
    }
    return true;
}
```



# Leetcode113. 路径总和||

使用前序遍历的思想，递归即可

代码：

```java
public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> ans = new ArrayList<>();
    if (root == null)
        return ans;
    List<Integer> path = new ArrayList<>();
    path.add(root.val);
    dfs(root, root.val, targetSum, path, ans);
    return ans;
}

private void dfs(TreeNode root, int sum, int targetSum, List<Integer> path, List<List<Integer>> ans) {
    if (root.left == null && root.right == null) {
        if (sum == targetSum) {
            ans.add(new ArrayList<>(path));
        }
        return;
    }
    
    if (root.left != null) {
        path.add(root.left.val);
        dfs(root.left, sum + root.left.val, targetSum, path, ans);
        path.remove(path.size() - 1);
    }
    
    if (root.right != null) {
        path.add(root.right.val);
        dfs(root.right, sum + root.right.val, targetSum, path, ans);
        path.remove(path.size() - 1);
    }
}
```



# Leetcode129. 求根节点到叶子结点数字之和

使用递归，还是前序遍历的思想

代码：

```java
private int ans = 0;

public int sumNumbers(TreeNode root) {
    if (root == null) return 0;
    dfs(root, root.val);
    return ans;
}

private void dfs(TreeNode root, int nowSum) {
    if (root.left == null && root.right == null) {
        ans += nowSum;
    }
    
    if (root.left != null) {
        dfs(root.left, nowSum * 10 + root.left.val);
    }
    
    if (root.right != null) {
        dfs(root.right, nowSum * 10 + root.right.val);
    }
}
```



# Leetcode43. 字符串相乘

利用模拟竖式乘法来做

需要另外写一个方法：计算两个String 的和

代码：

```java
public String multiply(String num1, String num2) {
    if ("0".equals(num1) || "0".equals(num2))
        return "0";
    String ans = "0";
    int m = num1.length(), n = num2.length();
    for (int i = n - 1; i >= 0; i--) {
        // 这里求得的是反向字符串
        StringBuilder cur = new StringBuilder();
        int t = 0;
        // 设置前导 0
        for (int j = n - 1; j > i; j--) {
            cur.append(0);
        }
        // 计算一位乘以另一个字符串
        int y = num2.charAt(i) - '0';
        for (int j = m - 1; j >= 0; j--) {
            int product = y * (num1.charAt(j) - '0') + t;
            cur.append(product % 10);
            t = product / 10;
        }
        while (t != 0) {
            cur.append(t % 10);
            t /= 10;
        }
        // 之所以要 reverse 是因为参数必须是正序的数字，而我们之前算出来的 String 是反向的
        ans = add(ans, cur.reverse().toString());
    }
    return ans;
}

public String add(String num1, String num2) {
    int i = num1.length() - 1, j = num2.length() - 1, t = 0;
    StringBuilder ans = new StringBuilder();
    while (i >= 0 || j >= 0 || t != 0) {
        int x = 0, y = 0;
        if (i >= 0)
            x = num1.charAt(i) - '0';
        if (j >= 0)
            y = num2.charAt(j) - '0';
        int res = x + y + t;
        ans.append(res % 10);
        t = res / 10;
        i--;
        j--;
    }
    ans.reverse();
    return ans.toString();
}
```



# Leetcode151. 翻转字符串中的单词

使用双指针算法，一个指针指向最后一个字符然后向前滑动。

代码：

```java
public String reverseWords(String s) {
    StringBuilder sb = new StringBuilder();
    boolean isFirst = true;
    for (int i = s.length() - 1; i >= 0; i--) {
        if (s.charAt(i) == ' ') continue;
        int j = i;
        for (j = i; j >= 0; j--) {
            if (s.charAt(j) == ' ')
                break;
        }
        if (isFirst) {
            sb.append(s, j + 1, i + 1);
            isFirst = false;
        } else {
            sb.append(" ").append(s, j + 1, i + 1);
        }
        i = j;
    }
    return sb.toString();
}
```



# Leetcode543. 二叉树的直径

思路：使用DFS，depth(root) 函数表示以该节点为根节点的子树的最大深度为多少

这样直径就等于 `max(depth(left) + depth(right))`

代码：

```java
private int ans = 0;

public int diameterOfBinaryTree(TreeNode root) {
    depth(root);
    return ans;
}

private int depth(TreeNode root) {
    if (root == null) {
        return 0;
    }
    int lDepth = depth(root.left);
    int rDepth = depth(root.right);
    ans = Math.max(ans, lDepth + rDepth);
    
    return Math.max(lDepth, rDepth) + 1;
}
```



# Leetcode128. 最长连续序列

思路：使用set集合去重，然后从集合中开始枚举

枚举思路：

只枚举连续序列中最小的数，即如果出现`num - 1` 在序列中出现，则跳过

否则查看 `num + 1` 的数是否存在于set中，若存在，则当前长度加1，继续查看更大的数是否在序列中

代码：

```java
public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        set.add(num);
    }
    
    int maxSize = 0;
    
    for (Integer num : set) {
        if (!set.contains(num - 1)) {
            int currentNum = num;
            int curSize = 1;
            while (set.contains(currentNum + 1)) {
                currentNum ++;
                curSize ++;
            }
            maxSize = Math.max(maxSize, curSize);
        }
    }
    
    return maxSize;
}
```



# Leetcode162. 寻找峰值

思路：

题目要求O(logn) 时间复杂度解决问题，因此由此想到二分

考虑一种情况：

中间元素小于右相邻元素，那么右相邻元素可能是峰值，因为它至少满足了大于左相邻元素

- 继续向右，如果右相邻元素大于它的右相邻元素，那么这个元素一定是峰值，因为我们之前已经保证这个右相邻元素大于它的左相邻元素。
- 如果右相邻元素小于它的右相邻元素，则继续向右，最坏情况是直到最后一位，最后一个数字它大于它的左相邻元素，而它的右相邻元素又是负无穷，因此最后一个元素就是峰值。

左相邻元素同理。

代码：

```java
public int findPeakElement(int[] nums) {
    if (nums == null || nums.length <= 0)
        return -1;
    int l = 0, r = nums.length - 1;
    
    while (l <= r) {
        int mid = l + r >> 1;
        if (mid - 1 >= 0 && nums[mid - 1] > nums[mid]) {
            r = mid - 1;
        } else if (mid + 1 < nums.length && nums[mid + 1] > nums[mid]) {
            l = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}
```

# Leetcode110. 平衡二叉树

这题写一个深搜函数：`getDepth(root)`，得到节点的子树高度，同时判断，如果左子树和右子树的差值的绝对值大于1，那么结果为false。

代码：

```java
private boolean ans = true;

public boolean isBalanced(TreeNode root) {
    getDepth(root);
    return ans;
}

private int getDepth(TreeNode root) {
    if (root == null) return 0;
    
    int left = getDepth(root.left);
    int right = getDepth(root.right);
    
    ans = ans && (Math.abs(left - right) <= 1);
    
    return Math.max(left, right) + 1;
}
```



# Leetcode138. 复制带随机指针的链表

思路，先用map将旧节点和新节点的映射保存下来

之后再遍历一遍确定新节点的random的指向

代码：

```java
public Node copyRandomList(Node head) {
    Map<Node, Node> map = new HashMap<>();
    
    Node cur = head;
    Node dummy = new Node(-1);
    Node p = dummy;
    while (cur != null) {
        Node node = new Node(cur.val);
        p.next = node;
        p = p.next;
        map.put(cur, node);
        cur = cur.next;
    }
    
    cur = head;
    p = dummy.next;
    while (cur != null) {
        if (cur.random == null)
            p.random = null;
        else
            p.random = map.get(cur.random);
        p = p.next;
        cur = cur.next;
    }
    
    return dummy.next;
}
```

# LeetCode. 求根节点到叶节点数字之和

使用深度优先搜索

代码：

```java
private int ans = 0;

public int sumNumbers(TreeNode root) {
    if (root == null) return 0;
    dfs(root, root.val);
    return ans;
}

private void dfs(TreeNode root, int val) {
    if (root.left == null && root.right == null) {
        ans += val;
        return;
    }
    if (root.left != null) 
        dfs(root.left, val * 10 + root.left.val);
    
    if (root.right != null)
        dfs(root.right, val * 10 + root.right.val);
}
```



# Leetcode322. 零钱兑换

![image-20220401121253921](C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220401121253921.png)

使用动态规划

代码：

```java
public int coinChange(int[] coins, int amount) {
    if (amount == 0)
        return 0;
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, Integer.MAX_VALUE);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        int min = Integer.MAX_VALUE - 1;
        
        for (int coin : coins) {
            if (i - coin >= 0) {
                min = Math.min(min, dp[i - coin]);
            }
        }
        dp[i] = min + 1;
    }
    
    if (dp[amount] == Integer.MAX_VALUE) return -1;
    return dp[amount];
}
```

# Leetcode221. 最大正方形

思路：使用动态规划

`dp[i][j] = x` 表示右下角以 (i, j) 结尾的全为 '1' 的正方形的最大边长

状态转移方程：

```
dp[i][j] = Math.min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1;
```

代码：

```java
public int maximalSquare(char[][] matrix) {
    int maxSize = 0;
    int n = matrix.length;
    if (n <= 0) return 0;
    int m = matrix[0].length;
    
    int[][] dp = new int[n][m];
    // 初始化
    for (int i = 0; i < n; i++) {
        dp[i][0] = matrix[i][0] == '1' ? 1 : 0;
    }
    for (int i = 0; i < m; i++) {
        dp[0][i] = matrix[0][i] == '1' ? 1 : 0;
    }
    
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            if (matrix[i][j] == '1') {
                // 状态转移
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1;
            } else {
                dp[i][j] = 0;
            }
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            maxSize = Math.max(maxSize, dp[i][j]);
        }
    }
    
    return maxSize * maxSize;
}
private int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}
```



# Leetcode165. 比较版本号

思路就是首先按照 "." 分割字符串，然后使用双指针比较即可，最后再根据指针指向的位置最后再扫个尾即可。

代码：

```java
public int compareVersion(String version1, String version2) {
    if (version1 == null || version2 == null) return 0;
    String[] split1 = version1.split("\\.");
    String[] split2 = version2.split("\\.");
    int i = 0, j = 0;
    
    while (i < split1.length && j < split2.length) {
        int i1 = Integer.parseInt(split1[i]);
        int j1 = Integer.parseInt(split2[j]);
        if (i1 < j1) return -1;
        else if (i1 > j1) return 1;
        i++; j++;
    }
    if (i == split1.length) {
        for (int k = j; k < split2.length; k++) {
            if (Integer.parseInt(split2[k]) > 0) return -1;
        }
    }
    if (j == split2.length) {
        for (int k = i; k < split1.length; k++) {
            if (Integer.parseInt(split1[k]) > 0) return 1;
        }
    }
    return 0;
}
```



# Leetcode122. 买卖股票的最佳时机

这题可以用动态规划也可以用贪心

## 动态规划

状态表示：

```java
dp[i][0] = x  // 表示第 i 天结束交易时不持有股票时的最大收益是 x
dp[i][1] = x  // 表示第 i 天结束交易时  持有股票时的最大收益是 x
```

转移方程：

```java
// 第 i 天交易结束不持有股票，可能是第 i - 1 天就不持有股票，第 i 天当然就是第 i - 1 天的收益
// 也有可能是第 i - 1天结束后持有股票(第 i 天的股票)，然后再卖出
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + price[i]);
// 第 i 天交易结束持有股票，可能是第 i - 1 天就持有股票，第 i 天不作处理
// 也有可能是第 i - 1 天不持有股票，然后购买第 i 天的股票
dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - price[i]);
```

代码：

```java
public int maxProfit(int[] prices) {
    if (prices == null) return 0;
    int n = prices.length;
    if (n <= 1) return 0;
    // dp[i][0] 表示第 i 天交易完手里没有股票的情况， dp[i][1] 表示第 i 天交易完手里有股票的情况
    int[][] dp = new int[n][2];
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for (int i = 1; i < n; i++) {
        // 第 i 天不购买股票                 第 i 天卖出第 i 天购买的股票
        dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        dp[i][1] = Math.max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
    }
    return dp[n - 1][0];
}
```

## 贪心

如果后一天的股票价格高于之前的股票价格，则直接赚取差价。

代码：

```java
public int maxProfit(int[] prices) {
    if (prices == null) return 0;
    int n = prices.length;
    if (n <= 1) return 0;
    int ans = 0;
    for (int i = 1; i < n; i++) {
        ans += Math.max(0, prices[i] - prices[i - 1]);
    }
    return ans;
}
```



# Leetcode209. 长度最小的子数组

## 滑动窗口

使用滑动窗口算法，每次添加右边的元素进入窗口，如果窗口总和大于 target，则收缩窗口，同时更新答案。

代码：

```java
public int minSubArrayLen(int target, int[] nums) {
    if (nums == null) return 0;
    int n = nums.length;
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        if (target == nums[0]) return 1;
        else return 0;
    }
    
    int l = 0, r = 0;
    int curSum = 0;
    int res = 0x3f3f3f3f;
    while (r < n) {
        curSum += nums[r];
        r ++;
        while (curSum >= target) {
            curSum -= nums[l];
            res = Math.min(r - l, res);
            l ++;
        }
    }
    if (res == 0x3f3f3f3f) return 0;
    return res;
}
```

# Leetcode958. 二叉树的完全性检验

使用 BFS，自己构造一个数据结构 Node，存储树节点在数组中的下标，从 1 开始
它的左节点是 2 * i，右节点是 2 * i + 1 （父节点的下标是 i ）

最后只要判断一下数组的最后一个节点与数组的大小关系即可

只要不相等则一定不是完全二叉树

代码：

```java
public boolean isCompleteTree(TreeNode root) {
    List<MyNode> list = new ArrayList<>();
    list.add(new MyNode(1, root));
    int i = 0;
    while (i < list.size()) {
        MyNode myNode = list.get(i++);
        TreeNode node = myNode.node;
        if (node.left != null) {
            list.add(new MyNode(2 * myNode.code, node.left));
        }
        if (node.right != null) {
            list.add(new MyNode(2 * myNode.code + 1, node.right));
        }
    }
    return list.get(list.size() - 1).code == i;
}

static class MyNode {
    int code;
    TreeNode node;
    MyNode() {};
    MyNode(int code, TreeNode node) {
        this.code = code;
        this.node = node;
    }
}
```

# Leetcode48. 旋转图像

这题要求O(1) 空间复杂度实现

旋转图像各个角度的通用O(1)解题思路：

- 顺时针旋转90：先沿横中轴线翻转，再沿主对角线翻转矩阵；

- 顺时针旋转180：先沿横中轴线翻转矩阵，再沿竖中轴线翻转矩阵；

- 顺时针旋转270：先沿竖中轴线翻转矩阵，再沿主对角线线翻转矩阵；

代码：

```java
public void rotate(int[][] matrix) {
    // 先水平翻转，再按照主对角线翻转
    if (matrix == null || matrix.length <= 0 || matrix[0].length <= 0) return;
    int n = matrix.length, m = matrix[0].length;
    // 横中轴线
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < m; j++) {
            swap(matrix, i, j, n - i - 1, j);
        }
    }
    // 主对角线翻转
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < m; j++) {
            swap(matrix, i, j, j, i);
        }
    }
}

public void swap(int[][] matrix, int row1, int col1, int row2, int col2) {
    int temp = matrix[row1][col1];
    matrix[row1][col1] = matrix[row2][col2];
    matrix[row2][col2] = temp;
}
```

# Leetcode394. 字符串解码



代码：

````java
public String decodeString(String s) {
    return recurDecode(s, 0, s.length());
}

public String recurDecode(String s, int index, int end) {
    StringBuilder res = new StringBuilder();
    for (int i = index; i < end; i++) {
        if (s.charAt(i) <= '9' && s.charAt(i) >= '0') {
            int j = i;
            while (s.charAt(j) <= '9' && s.charAt(j) >= '0') j ++;
            
            // 得到这个括号的结尾
            Stack<Character> stack = new Stack<>();
            stack.push(s.charAt(j));
            int cursor = j + 1;
            while (!stack.isEmpty()) {
                if (s.charAt(cursor) == '[') stack.push(s.charAt(cursor))
                else if (s.charAt(cursor) == ']') stack.pop();
                cursor ++;
            }
            
            int num = Integer.parseInt(s.substring(i, j));
            for (int k = 0; k < num - 1; k++) {  // 这里要 num - 1 因为后面还有字符串需要 append
                // 递归
                res.append(recurDecode(s, j + 1, cursor));
            }
            i = j - 1;  // 调整指针
        } else if (s.charAt(i) >= 'a' && s.charAt(i) <= 'z'){
            res.append(s.charAt(i));
        }
    }
    return res.toString();
}
````

# 原创. 圆环回原点问题

圆环上有10个点，编号为0~9。从0点出发，每次可以逆时针和顺时针走一步，问走n步回到0点共有多少种走法。

```
输入: 2
输出: 2
解释：有2种方案。分别是0->1->0和0->9->0
```

这题类似于LeetCode70. 爬楼梯，使用动态规划算法

**状态表示：**

````c
dp[i][j] = x // 表示走 i 步到 j 点的路径数量为 x
````

**初始化：**

```c
dp[0][0] = 1
```

**状态转移：**

```java
// 从 i - 1 步到第 i 步 只差一步，且一定要在 j 的左右，即 j - 1 或者 j + 1
dp[i][j] = dp[i - 1][(j - 1 + length) % length] + dp[i - 1][(j + 1 + length) % length];
// 这里要(j - 1 + length) % length 是因为可能会超过 length 或者小于 0
```

代码：

```java
static int backToOrigin(int n) {
    int length = 10;
    int[][] dp = new int[n + 1][length + 1];
    
    dp[0][0] = 1;
    
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= length; j++) {
            dp[i][j] = dp[i - 1][(j - 1 + length) % length] + dp[i - 1][(j + 1) % length];
        }
    }
    
    return dp[n][0];
}
```

# 原创. 36进制加法

要求不能将36进制转为10进制再相加，必须按照36进制相加

这题类似于大数相加，从两个字符串的尾部逐位相加最后将结果反转即可。

代码：

```java
static String add36(String num1, String num2) {
    StringBuilder res = new StringBuilder();
    if (num1 == null) return num2;
    if (num2 == null) return num1;
    
    int n1 = num1.length(), n2 = num2.length();
    
    if (n1 == 0) return num2;
    if (n2 == 0) return num1;
    int i = num1.length() - 1, j = num2.length() - 1;
    int t = 0;
    
    while (i >= 0 && j >= 0) {
        char c1 = num1.charAt(i), c2 = num2.charAt(j);
        int number1 = (c1 <= '9' && c1 >= '0') ? (c1 - '0') : (c1 - 'a' + 10),
            number2 = (c2 <= '9' && c2 >= '0') ? (c2 - '0') : (c2 - 'a' + 10);
        int sum = (number1 + number2 + t) % 36;
        t = (number1 + number2 + t) / 36;
        res.append((char) ((sum <= 9 && sum >= 0) ? ('0' + sum) : ('a' + sum - 10)));
        i --;
        j --;
    }
    
    while (i >= 0) {
        char c1 = num1.charAt(i);
        int number = (c1 <= '9' && c1 >= '0') ? (c1 - '0') : (c1 - 'a' + 10);
        int sum = (number + t) % 36;
        t = (number + t) / 36;
        res.append((char) ((sum <= 9 && sum >= 0) ? ('0' + sum) : ('a' + sum - 10)));
        i --;
    }
    
    while (j >= 0) {
        char c1 = num2.charAt(j);
        int number = (c1 <= '9' && c1 >= '0') ? (c1 - '0') : (c1 - 'a' + 10);
        int sum = (number + t) % 36;
        t = (number + t) / 36;
        res.append((char) ((sum <= 9 && sum >= 0) ? ('0' + sum) : ('a' + sum - 10)));
        j --;
    }
    
    if (t > 0) {
        res.append(t);
    }
    
    return res.reverse().toString();
}
```

# Leetcode31. 下一个排列

这题就是模拟题，首先找到右边第一个非降序的下标，例如：

```c
[6, 8, 1, 9, 5]
```

中的 1，下标为 2

再找到从右边数第一个大于 1 的数的下标，即 5 的下标为 4

将这两个下标对应的数字交换：

```c
[6, 8, 5, 9, 1]
```

将[2 + 1, 5 - 1] 范围内的数字反转可得：

```c
[6, 8, 5, 1, 9]
```

即下一个排列

代码：

```java
public void nextPermutation(int[] nums) {
    if (nums == null) return;
    int n = nums.length;
    if (n <= 1) return;
    // 第一遍找到右边较小数。
    int i;
    for (i = n - 2; i >= 0; i--) {
        if (nums[i] < nums[i + 1]) {
            break;
        }
    }
    // 如果不是单调递减
    if (i >= 0) {
        int j;
        // 找到右边第一个大于之前找到的较小数的数
        for (j = n - 1; j >= 0; j--) {
            if (nums[j] > nums[i]) {
                break;
            }
        }
        // 将两个数交换
        swap(nums, i, j);
    }
    reverse(nums, i + 1, n - 1);
}

public void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}

public void reverse(int[] nums, int l, int r) {
    while (l < r) {
        int temp = nums[l];
        nums[l] = nums[r];
        nums[r] = temp;
        l++; r--;
    }

}
```

# Leetcode892. 和至少为 K 的子数组

这题和 LeetCode209 类似，区别就是这里的数组中允许负数出现。

这题使用前缀和 + 滑动窗口算法实现。

用 sum[k] 表示前缀和，那么如果 `sum[y] - sum[x] >= k` 那么这个子数组就是符合题目子数组。

那么对于每个 y 找出满足 `sum[y] - sum[x] >= k 的最大的 x` ，如果 y - x 比之前的长度要小就记录新的最小值。

维持队列的两个规则：

1. 对于新加入的 y，前面的 sum[x] 都要比新加入的 sum[y] 要小，比 sum[y] 大的 p[x] 都要 pop 掉，因为根据题目条件 `k > 0`

2. 当队列中的第一个（队头）的 x 满足 `sum[y] - sum[x] >= k`，第一个 x 可以被 pop 掉

   因此此时我们构成了一个 `sum[y] - sum[x] >= k`，之后这个 x 就没作用了，因为 y 会向右滑动，因此 `y' - x > y - x`，而我们需要的答案要求是最小值，因此队头的这个 x 可以直接 remove 掉。

代码：

```java
public int shortestSubarray(int[] nums, int k) {
    if (nums == null) return -1;
    int n = nums.length;
    if (n <= 0) return -1;
    if (n == 1) {
        if (nums[0] >= k) return 1;
        else return -1;
    }
    
    long[] sum = new long[n + 1];
    for (int i = 1; i <= n; i++) {
        sum[i] = sum[i - 1] + nums[i - 1];
    }
    
    int res = n + 1;
    Deque<Integer> deque = new LinkedList<>();
    for (int y = 0; y < sum.length; y++) {
        while (!deque.isEmpty() && sum[y] <= sum[deque.getLast()])
            deque.removeLast();
        while (!deque.isEmpty() && sum[y] >= sum[deque.getFirst()] + k)
            res = Math.min(res, y - deque.removeFirst());
        
        deque.addLast(y);
    }
    
    return (res < n + 1) ? res : -1;
}
```

# Leetcode23. 合并 K 个升序链表

有三种方法，分别是暴力法，即两两合并升序链表，一直合并n - 1次，时间复杂度为 n^2^；分治法，是暴力法的升级版，相当于归并；第三种是使用优先队列，将每条链表的第一个节点放入小根堆中，这样堆顶元素就是最小元素，我们将其接入到答案的链表中即可。

其中暴力法时间复杂度 n^2^，空间复杂度为O(1)

分治法时间复杂度O(n logn)，空间复杂度O(logn)

小根堆法时间复杂度O(n logn)，空间复杂度 O(n)

分治法：

```java
    public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int l, int r) {
        if (l == r) {
            return lists[l];
        }
        if (l > r) {
            return null;
        }
        int mid = l + r >> 1;
        return mergeTwoList(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    private ListNode mergeTwoList(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) return null;
        if (list1 == null || list2 == null) return list1 == null ? list2 : list1;

        ListNode dummy = new ListNode();
        ListNode cur = dummy;
        ListNode p = list1, q = list2;

        while (p != null && q != null) {
            if (p.val < q.val) {
                cur.next = p;
                p = p.next;
            } else {
                cur.next = q;
                q = q.next;
            }
            cur = cur.next;
        }

        if (p != null) cur.next = p;
        if (q != null) cur.next = q;

        return dummy.next;
    }
```

小根堆法：

```java
class Status implements Comparable<Status> {
    int val;
    ListNode pointer;
    @Override
    public int compareTo(Status status) {
        return this.val - status.val;
    }
    Status(int val, ListNode ptr) {
        this.val = val;
        this.pointer = ptr;
    }
}

private final PriorityQueue<Status> pile = new PriorityQueue<>();

public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
        return null;
    if (lists.length == 1)
        return lists[0];
    for (ListNode list : lists) {
        if (list != null)
            pile.offer(new Status(list.val, list));
    }
    ListNode dummy = new ListNode();
    ListNode cur = dummy;
    while (!pile.isEmpty()) {
        Status peek = pile.poll();
        cur.next = peek.pointer;
        cur = cur.next;
        if (peek.pointer.next != null) {
            pile.offer(new Status(peek.pointer.next.val, peek.pointer.next));
        }
    }
    return dummy.next;
}
```

# 原创题. 木头切割问题

**题目描述**

给定长度为n的数组，每个元素代表一个木头的长度，木头可以任意截断，从这堆木头中截出至少k个相同长度为m的木块。已知k，求max(m)。

输入两行，第一行n, k，第二行为数组序列。输出最大值。

> 输入
> 5 5
> 4 7 2 10 5
> 输出
> 4
> 解释：如图，最多可以把它分成5段长度为4的木头
>
> ![图片](https://mmbiz.qpic.cn/mmbiz_png/oD5ruyVxxVHVR60EJHyZEZAdt5KkTSSvpjP30ZWe9WxlFFHibiaPchmjVcVpkkCkVqUNicm9NReAvCbKC0vdy6sZg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

ps:数据保证有解，即结果至少是1。

这题使用二分算法，枚举[1, maxLen]，如果 mid 满足条件，由于这题要找最大的长度，所以可以直接切割掉[l, mid)，这就是标准的二分法

代码：

```java
private static boolean check(int[] arr, int k, int mid) {
    int ok = 0;
    
    for (int len : arr) {
        ok += len / mid;
    }
    
    return ok >= k;
}

public static void main(String[] args) {
    Scanner scan = new Scanner(System.in);
    int n = scan.nextInt(), k = scan.nextInt();
    int[] arr = new int[n];
    
    for (int i = 0; i < n; i++) {
        arr[i] = scan.nextInt();
    }
    int l = 0, r = -1;
    
    for (int num : arr) {
        r = Math.max(r, num);
    }
    
    int res = 0;
    while (l <= r) {
        int mid = l + r >> 1;
        if (check(arr, k, mid)) {
            l = mid + 1;
            res = Math.max(res, mid);
        } else {
            r = mid - 1;
        }
    }
    
    System.out.println(res);
}
```

# Leetcode662. 二叉树的最大宽度

这题和判断完全二叉树很像，手写一个类标注坐标，最后使用 BFS 算法即可。

代码：

```java
public int widthOfBinaryTree(TreeNode root) {
    Queue<Node> queue = new ArrayDeque<>();
    queue.add(new Node(root, 0L));
    int res = 1;
    while (!queue.isEmpty()) {
        int size = queue.size();
        long leftIndex = queue.peek().index;
        long rightIndex = 0L;
        for (int i = 0; i < size; i++) {
            Node node = queue.poll();
            if (node.node.left != null) {
                queue.add(new Node(node.node.left, node.index * 2 + 1));
            }
            if (node.node.right != null) {
                queue.add(new Node(node.node.right, node.index * 2 + 2));
            }
            if (i == size - 1) {
                // 本层的最后一个元素
                rightIndex = node.index;
            }
        }
        res = Math.max(res, (int) (rightIndex - leftIndex) + 1);
    }
    return res;
}

static class Node {
    TreeNode node;
    long index;
    Node() {}
    Node(TreeNode node, long index) {
        this.node = node;
        this.index = index;
    }
}
```

# Leetcode283. 移动零

智力题，用O(1) 空间复杂度，O(n) 时间复杂度实现了，只要定义一个 helper 变量帮助统计当前移动的非零元素的下标即可。

代码：

```java
public void moveZeroes(int[] nums) {
    if (nums == null || nums.length <= 1) return;
    int n = nums.length;
    int helper = 0;
    for (int i = 0; i < n; i++) {
        if (nums[i] != 0) {
            swap(nums, helper, i);
            helper ++;
        }
    }
}

private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}
```

# 剑指offer51. 数组中逆序对

这题使用归并的思想。

如图，我们将`{7, 5, 6, 4}`首先划分为:

![image-20220104095317967](C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220104095317967.png)

在合并`{5, 7}`和`{4, 6}`数组时

![image-20220104095209301](C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220104095209301.png)

如果P1指向的数大于P2指向的数，说明构成逆序对，那么逆序对数量加上第二个数组剩余的元素个数，然后P1左移。

如果P1指向的数小于P2指向的数，说明符合顺序，那么P2左移。

代码：

```java
public int reversePairs(int[] nums) {
    if (nums == null || nums.length <= 1) return 0;
    int n = nums.length;
    int[] copy = Arrays.copyOf(nums, n);
    return reversePairsCore(nums, copy, 0, n - 1);
}

public int reversePairsCore(int[] nums, int[] copy, int start, int end) {
    if (start == end) {
        copy[start] = nums[start];
        return 0;
    }
    
    int length = end - start >> 1;
    int left = reversePairsCore(copy, nums, start, start + length);  // 左边数组的逆序对个数
    int right = reversePairsCore(copy, nums, start + length + 1, end);  // 右边数组的逆序对个数
    
    int count = 0;  // 两个数组的逆序对个数
    int i = start + length;
    int j = end;
    int indexOfCopy = end;
    
    while (i >= start && j >= start + length + 1) {
        if (nums[i] > nums[j]) {
            copy[indexOfCopy--] = nums[i--];
            count += j - start - length;
        } else {
            copy[indexOfCopy--] = nums[j--];
        }
    }
    while (i >= start) {
        copy[indexOfCopy--] = nums[i--];
    }
    while (j >= start + length + 1) {
        copy[indexOfCopy--] = nums[j--];
    }
    return left + count + right;
}
```

# Leetcode315. 计算右侧小于当前元素的个数

这题和上一题 数组中逆序对 很像，都是用了归并算法，唯一的不同就是在合并的时候操作不同

这题同时采用了索引数组这一技巧，我们可以通过索引数组来找到某个数字在原来数组中的下标。

代码：

```java
List<Integer> ans = new ArrayList<>(); //记录最终的结果
int[] index; //原数组的索引数组，存储着原数组中每个元素对应的下标
int[] count; //记录题目中所求的count[i]

//入口
public List<Integer> countSmaller(int[] nums){
    int len = nums.length;
    index = new int[len];  // index[i] = x : 在目前 nums[i] 的那个元素，在原来数组的 x 位置
    count = new int[len];
    for (int i = 0; i < nums.length; i++) {
        index[i] = i;
    }
    MergeSort(nums, 0, nums.length - 1);
    for (int i = 0; i < len; i++) {
        ans.add(count[i]);
    }
    return ans;
}

/**
 * 归并排序
 */
private void MergeSort(int[] nums, int start, int end){
    if(start < end){
        int mid = start + (end - start) / 2;
        MergeSort(nums, start, mid); //将无序数组划分
        MergeSort(nums, mid + 1, end); //将无序数组划分
        merge(nums, start, mid, end); //再将两个有序数组合并,只不过这里要承接统计count[i]的功能
    }
}
/**
 *  双指针合并两个有序数组并统计count[i]
 */
private void merge(int[] nums, int start, int mid, int end){
    int P1 = start;
    int P2 = mid + 1;
    int cur = 0;
    int[] tmp = new int[end - start + 1]; //临时数组用于存储一次归并过程中排序好的元素，
    int[] tmpIndex = new int[end - start + 1];//临时数组的索引数组，存储这临时数组中每个元素对应的下标
    while (P1 <= mid && P2 <= end){
        if (nums[P1] > nums[P2]) {
            count[index[P1]] += end - P2 + 1;  // 更新答案数组
            tmp[cur] = nums[P1];
            tmpIndex[cur] = index[P1];
            P1 ++;
        } else {
            tmp[cur] = nums[P2];
            tmpIndex[cur] = index[P2];
            P2 ++;
        }
        cur ++;
    }
    while (P1 <= mid){
        tmp[cur] = nums[P1];
        tmpIndex[cur] = index[P1];
        P1++;
        cur++;
    }
    while (P2 <= end){
        tmp[cur] = nums[P2];
        tmpIndex[cur] = index[P2];
        P2++;
        cur++;
    }
    for (int i = 0; i < end - start + 1; i++) {
        nums[i + start] = tmp[i];
        index[i + start] = tmpIndex[i];
    }
}
```

# Leetcode778. 重构字符串

要求将一个字符串重构成为每个字符左右不能有相邻相同字符。例如 "aab" 是不成立的，而 "aba" 是成立的。

这题使用贪心算法解决，将字符串放入大根堆中，这个堆的排序规则是某个字符出现的次数越多就在上层。

```java
PriorityQueue<Character> qile = new PriorityQueue<>(new Comparator<Character>() {
    @Override
    public int compare(Character o1, Character o2) {
        return count[o2 - 'a'] - count[o1 - 'a'];
    }
});
```

每个从大根堆堆顶取走两个字符拼接

如果其中出现次数最多的字符的出现次数大于 `length + 1 >> 1` 那么源字符串不可能重构成为我们想要的字符串。

代码：

```java
public String reorganizeString(String s) {
    StringBuilder res = new StringBuilder();
    int[] count = new int[26];
    int maxLength = 0;
    for (int i = 0; i < s.length(); i++) {
        count[s.charAt(i) - 'a'] ++;
    }
    for (int i = 'a'; i <= 'z'; i++) {
        maxLength = Math.max(maxLength, count[i - 'a']);
    }
    if (maxLength > (s.length() + 1) / 2) return "";  // 不满足
    
    PriorityQueue<Character> qile = new PriorityQueue<>(new Comparator<Character>() {
        @Override
        public int compare(Character o1, Character o2) {
            return count[o2 - 'a'] - count[o1 - 'a'];
        }
    });  // 大根堆
    for (int i = 'a'; i <= 'z'; i++) {
        if (count[i - 'a'] > 0) qile.offer((char) i);
    }
    while (qile.size() > 1) {
        Character c1 = qile.poll();
        Character c2 = qile.poll();
        res.append(c1);
        res.append(c2);
        count[c1 - 'a'] --;
        count[c2 - 'a'] --;
        if (count[c1 - 'a'] > 0) {
            qile.offer(c1);
        }
        if (count[c2 - 'a'] > 0) {
            qile.offer(c2);
        }
    }
    if (qile.size() != 0) {
        res.append(qile.poll());
    }
    return res.toString();
}
```

# Leetcode739. 每日温度

这题使用单调栈算法，栈中存放原来数组的下标，当新元素进来时先于栈顶元素比较，如果新元素大于栈顶元素对应的数组元素，那么就将栈顶元素移除，同时更新答案数组。

代码：

```java
public int[] dailyTemperatures(int[] temperatures) {
    if (temperatures == null) return null;
    int n = temperatures.length;
    if (n == 0) return new int[0];
    if (n == 1) return new int[1];
    int[] res = new int[temperatures.length];
    Stack<Integer> stack = new Stack<>();  // 栈中存放逆序排列的数组元素的下标
    for (int i = 0; i < n; i++) {
        while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
            int num = stack.pop();
            res[num] = i - num;
        }
        stack.push(i);
    }
    return res;
}
```

# Leetcode106. 从中序和后序遍历确定二叉树

递归即可，根据后序数组的最后一个元素可以确定根节点的权值，再从中序遍历数组找到对应元素位置，那么这个位置左边的元素就是左子树元素，右边的元素就是右子树元素。

代码：

```java
private int[] inorder;
private int[] postorder;
private final Map<Integer, Integer> cache = new HashMap<>();

public TreeNode buildTree(int[] inorder, int[] postorder) {
    this.inorder = inorder;
    this.postorder = postorder;
    for (int i = 0; i < inorder.length; i++) {
        cache.put(inorder[i], i);
    }
    return builderCore(0, inorder.length - 1, 0, postorder.length - 1);
}

private TreeNode builderCore(int il, int ir, int pl, int pr) {
    if (il > ir || pl > pr) return null;
    int rootVal = postorder[pr];
    int rootIndex = cache.get(rootVal);
    
    TreeNode left = builderCore(il, rootIndex - 1, pl, pl + rootIndex - il - 1);
    TreeNode right = builderCore(rootIndex + 1, ir, pl + rootIndex - il, pr - 1);
    return new TreeNode(rootVal, left, right);
}
```

# Leetcode1143. 最长公共子序列

这题显然使用动态规划算法，二维数组表示状态

```
dp[i][j] = x 表示 s1 前 i 个字符与 s2 前 j 个字符的最长公共子序列长度为 x
```

转移方程：

```java
			dp[i - 1][j - 1] + 1,  s1.charAt(i) == s2.charAt(j);
dp[i][j] = 
    		Math.max(dp[i - 1][j], dp[i][j - 1]),  s1.charAt(i) != s2.charAt(j);
```

如果要求打印路径：

```java
StringBuilder res = new StringBuilder(dp[s1.length() - 1][s2.length() - 1]);
for (int i = s1.length(); i >= 1; i--) {
    for (int j = s2.length(); j >= 1; j--) {
        if (dp[i][j] > dp[i - 1][j] && dp[i][j] > dp[i][j - 1]) {
            res.append(s1.charAt(i - 1));
        }
    }
}
System.out.println(res.reverse());
```

代码：

```java
public int longestCommonSubsequence(String s1, String s2) {
    int[][] dp = new int[s1.length() + 1][s2.length() + 1];
    
    for (int i = 1; i <= s1.length(); i++) {
        for (int j = 1; j <= s2.length(); j++) {
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) 
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else 
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    
    return dp[s1.length()][s2.length()];
}
```

# Leetcode64. 最小路径和

这题仍然用动态规划求解

```c
dp[i][j] = x  // 表示走到 grid[i][j] 的最小路径和
```

那么最后返回的结果很显然，就是 `dp[n - 1][m - 1]`

初始化：

```java
for (int i = 0; i < n; i++) {
    // 初始化第一列
    dp[i][0] = i > 0 ? (dp[i - 1][0] + grid[i][0]) : grid[i][0];
}

for (int i = 0; i < m; i++) {
    // 初始化第一行
    dp[0][i] = i > 0 ? (dp[0][i - 1] + grid[0][i]) : grid[0][i];
}
```



状态转移方程：

```java
dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
```

代码：

```java
public int minPathSum(int[][] grid) {
    int n = grid.length, m = grid[0].length;
    int[][] dp = new int[n][m];
    
    for (int i = 0; i < n; i++) {
        dp[i][0] = i > 0 ? (dp[i - 1][0] + grid[i][0]) : grid[i][0];
    }
    for (int i = 0; i < m; i++) {
        dp[0][i] = i > 0 ? (dp[0][i - 1] + grid[0][i]) : grid[0][i];
    }
    
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }
    
    return dp[n - 1][m - 1];
}
```

如果要打印路径：

```java
Deque<Integer> res = new LinkedList<>();
int i = n - 1, j = m - 1;
while (i >= 0 && j >= 0) {
    if (i - 1 >= 0 && dp[i - 1][j] + grid[i][j] == dp[i][j]) {
        res.addFirst(grid[i][j]);
        i --;
    } else if (j - 1 >= 0 && dp[i][j - 1] + grid[i][j] == dp[i][j]) {
        res.addFirst(grid[i][j]);
        j --;
    } else {
        break;
    }
}

res.addFirst(grid[0][0]);  // 添加起点

System.out.println(res);
```

# Leetcode169. 多数元素

从一个数组中找出出现次数大于 n / 2 的元素

这题可以排序数组，可以用哈希表解决，也可以排序数组解决，但是一个空间复杂度O(n)，一个时间复杂度 O(nlogn)，都不符合要求。

这里使用摩尔投票算法。

每次维护一个候选人，一个候选票。

如果遍历的元素是候选人，则投票数加一，否则投票数减一，如果投票数等于0，那么候选人就是当前元素。

代码：

```java
public int majorityElement(int[] nums) {
    Integer candidate = null;
    int count = 0;
    for (int num : nums) {
        if (count == 0) {
            candidate = num;
        }
        count += (num == candidate) ? 1 : -1;
    }
    return candidate == null ? -1 : candidate;
}
```

# Leetcode718. 最长重复子数组

这题用 DP 算法

**状态表示：**

```java
dp[i][j] = x;  // 表示 nums1 前 i 个元素，nums2 前 j 个元素结尾的最长子数组的长度为 x
```

**转移方程：**

```java
			dp[i - 1][j - 1] + 1,  // nums1[i - 1] = nums2[j - 1]
dp[i][j] = 
    		0,   // nums1[i - 1] != nums2[j - 1]
```

代码：

```java
public int findLength(int[] nums1, int[] nums2) {
    int n = nums1.length, m = nums2.length;
    int[][] dp = new int[n + 1][m + 1];
    int res = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = 0;
            }
            res = Math.max(res, dp[i][j]);
        }
    }
    return res;
}
```

# Leetcode226. 翻转二叉树

这题使用深搜（相当于前序遍历）

![image-20220422095636150](C:\Users\HASEE\Desktop\读书笔记\CodeTops.assets\image-20220422095636150.png)

代码：

```java
public TreeNode invertTree(TreeNode root) {
    dfs(root);
    return root;
}

private void dfs(TreeNode root) {
    if (root == null) return;
    if (root.left == null && root.right == null) return;
    
    TreeNode temp = root.left;
    root.left = root.right;
    root.right = temp;
    
    dfs(root.left);
    dfs(root.right);
}
```

# Leetcode34. 在排序数组中查找元素的第一个位置和最后一个位置

这题使用二分算法，一个二分负责找右边界，一个二分负责找左边界

找右边界：

```java
while (l < r) {
    // 找右边界
    int mid = l + r + 1 >> 1;
    if (nums[mid] <= target) {
        l = mid;
    } else {
        r = mid - 1;
    }
}
rightBound = l;
```

找左边界：

```java
while (l < r) {
    // 找左边界
    int mid = l + r >> 1;
    if (nums[mid] >= target) {
        r = mid;
    } else {
        l = mid + 1;
    }
}
leftBound = l;
```

代码：

```java
public int[] searchRange(int[] nums, int target) {
    int n = nums.length;
    if (n == 0) return new int[]{-1, -1};
    int l = 0, r = n - 1;
    int leftBound, rightBound;
    
    while (l < r) {
        // 找右边界
        int mid = l + r + 1 >> 1;
        if (nums[mid] <= target) {
            l = mid;
        } else {
            r = mid - 1;
        }
    }
    rightBound = l;
    if (nums[rightBound] != target) return new int[]{-1, -1};
    
    l = 0; r = n - 1;
    while (l < r) {
        // 找左边界
        int mid = l + r >> 1;
        if (nums[mid] >= target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    leftBound = l;
    if (nums[leftBound] != target) return new int[]{-1, -1};
    
    return new int[]{leftBound, rightBound};
}
```

# Leetcode14. 最长公共前缀

这题我用的纵向比较，从第一个字符开始，不过有一个优化，就是当之前的字符串比较最小长度的最长公共前缀时，我们就可以不用完全扫描整个字符串，只用扫描这个最小长度之前的即可。

代码：

```java
public String longestCommonPrefix(String[] strs) {
    int n = strs.length;
    if (n <= 0) return "";
    if (n <= 1) return strs[0];
    int length = helper(strs[0], strs[1], 210);
    for (int i = 2; i < n; i++) {
        length = Math.min(length, helper(strs[i - 1], strs[i], length));
    }
    return strs[0].substring(0, length);
}

// 得到两个字符串最长公共前缀长度
private int helper(String s1, String s2, int length) {
    int index = 0;
    for (index = 0; index < min(s1.length(), s2.length(), length); index++) {
        if (s1.charAt(index) != s2.charAt(index)) return index;
    }
    return index;
}

private int min(int a, int b, int c) {
    return Math.min(Math.min(a, b), c);
}
```

